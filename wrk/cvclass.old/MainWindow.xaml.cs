using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Data;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Navigation;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;

// To learn more about WinUI, the WinUI project structure,
// and more about our project templates, see: http://aka.ms/winui-project-info.

//　https://huggingface.co/onnx
//　NETSDK1083 fixit
//　https://zenn.dev/shinta0806/articles/dotnet8-netsdk1083

namespace cvclass
{
	/// <summary>
	/// An empty window that can be used on its own or navigated to within a Frame.
	/// </summary>
	public sealed partial class MainWindow : Window
	{
		public MainWindow()
		{
			this.InitializeComponent();
		}

		private void Run_Click(object sender, RoutedEventArgs e)
		{
			//var pModelFilepath = "D:\\Home\\Rink\\projects\\assets\\networks\\fcn-resnet50-12-int8.onnx";
			//var pModelFilepath = "D:\\Home\\Datas\\Network\\SSD-MobilenetV1\\ssd_mobilenet_v1_13-qdq.onnx";	//　uint8
			//var pModelFilepath = "D:\\Home\\Datas\\Network\\SSD-MobilenetV1\\ssd_mobilenet_v1_12.onnx"; //　uint8
			//var pModelFilepath = "D:\\Home\\Datas\\Network\\SSD\\ssd-12.onnx";
			var pModelFilepath = "D:\\Home\\rink\\projects\\assets\\networks\\ssd-12.onnx";
			var pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\01.jpg";
			var pModel = new SSDMobileNet();
			if (pModel.Run(pModelFilepath, pImageFilepath) == false)
			{
				;
			}
		}
	}
}
