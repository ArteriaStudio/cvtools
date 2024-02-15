using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.Media;
using Windows.Storage.Streams;
using Windows.Storage;
using System.Threading;
using Windows.Foundation;

namespace cvclass
{
	public class CVBase
	{
		protected static LearningModel			m_pModel = null;
		protected static LearningModelSession	m_pSession = null;

		public static bool LoadModel(string pModelFilepath)
		{
			int ticks = Environment.TickCount;
			m_pModel = LearningModel.LoadFromFilePath(pModelFilepath);
			ticks = Environment.TickCount - ticks;
			Console.WriteLine($"model file loaded in {ticks} ticks");

			return(true);
		}

		public static bool CreateSession()
		{
			// Create the evaluation session with the model and device
			m_pSession = new LearningModelSession(m_pModel, new LearningModelDevice(LearningModelDeviceKind.Default));
			return(true);
		}

		//　※ サンプルまま
		public static ColorManagementMode GetColorManagementMode()
		{
			// Get model color space gamma
			string gammaSpace = "";
			bool doesModelContainGammaSpaceMetadata = m_pModel.Metadata.TryGetValue("Image.ColorSpaceGamma", out gammaSpace);
			if (!doesModelContainGammaSpaceMetadata)
			{
				;
			}
			if (!doesModelContainGammaSpaceMetadata || gammaSpace.Equals("SRGB", StringComparison.CurrentCultureIgnoreCase))
			{
				return ColorManagementMode.ColorManageToSRgb;
			}
			// Due diligence should be done to make sure that the input image is within the model's colorspace. There are multiple non-sRGB color spaces.
			return ColorManagementMode.DoNotColorManage;
		}

		//　pImageFilepath：画像ファイルパス
		//　※ サンプルまま
		public static ImageFeatureValue LoadImageFile(ColorManagementMode colorManagementMode, string pImageFilepath)
		{
			BitmapDecoder decoder = null;
			try
			{
				StorageFile imageFile = AsyncHelper(StorageFile.GetFileFromPathAsync(System.IO.Path.GetFullPath(pImageFilepath)));
				IRandomAccessStream stream = AsyncHelper(imageFile.OpenReadAsync());
				decoder = AsyncHelper(BitmapDecoder.CreateAsync(stream));
			}
			catch (Exception e)
			{
				System.Environment.Exit(e.HResult);
			}
			SoftwareBitmap softwareBitmap = null;
			try
			{
				softwareBitmap = AsyncHelper(
					decoder.GetSoftwareBitmapAsync(
						decoder.BitmapPixelFormat,
						decoder.BitmapAlphaMode,
						new BitmapTransform(),
						ExifOrientationMode.RespectExifOrientation,
						colorManagementMode
					)
				);
			}
			catch (Exception e)
			{
				System.Environment.Exit(e.HResult);
			}
			softwareBitmap = SoftwareBitmap.Convert(softwareBitmap, BitmapPixelFormat.Bgra8, BitmapAlphaMode.Premultiplied);
			VideoFrame inputImage = VideoFrame.CreateWithSoftwareBitmap(softwareBitmap);
			return ImageFeatureValue.CreateFromVideoFrame(inputImage);
		}

		private static T AsyncHelper<T>(IAsyncOperation<T> operation)
		{
			AutoResetEvent waitHandle = new AutoResetEvent(false);
			operation.Completed = new AsyncOperationCompletedHandler<T>((op, status) =>
			{
				waitHandle.Set();
			});
			waitHandle.WaitOne();
			return operation.GetResults();
		}
	}
}
