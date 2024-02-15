using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.System;

namespace cvclass
{
	//　Fully Convolutional Network 処理モジュール
	//（共通理解）
	//　まず機械学習結果のネットワークモデルは、入力と出力の様式に統一的な規格は存在しない。
	//　入力アセンブリと出力フェッチは、モデル別に実装する必要があることがほぼ約束されている。
	//　なので温和しくモデル別に実装してよろしい。
	public class FCN : CVBase
	{
		//　画像分類と領域区分を実行
		//（引数）
		//　pModelFilepath：モデルファイルのパス
		//　pImageFilepath：画像ファイルパス
		//（備考）
		//　引数の通り、入力画像はストレージ上のファイルであることを前提とする。
		public static bool Run(string pModelFilepath, string pImageFilepath)
		{
			if (LoadModel(pModelFilepath) == false)
			{
				return (false);
			}
			ColorManagementMode pColorManagementMode = GetColorManagementMode();
			ImageFeatureValue imageTensor = LoadImageFile(pColorManagementMode, pImageFilepath);

			if (CreateSession() == false)
			{
				return(false);
			}

			LearningModelBinding binding = new LearningModelBinding(m_pSession);
			binding.Bind(m_pModel.InputFeatures.ElementAt(0).Name, imageTensor);

			var ticks = Environment.TickCount;
			var pResults = m_pSession.Evaluate(binding, "RunId");
			ticks = Environment.TickCount - ticks;
			Console.WriteLine($"model run took {ticks} ticks");

			//　結果をフェッチ
			PrintResults(pResults);

			return (true);
		}

		//　
		private static void LoadLabels()
		{
			/*
			// Parse labels from label json file.  We know the file's 
			// entries are already sorted in order.
			var fileString = File.ReadAllText(_labelsFileName);
			var fileDict = JsonConvert.DeserializeObject<Dictionary<string, string>>(fileString);
			foreach (var kvp in fileDict)
			{
				_labels.Add(kvp.Value);
			}
			*/
		}

		//　
		private static void PrintResults(LearningModelEvaluationResult pResults)
		{
			// load the labels
			LoadLabels();

			//　pythonのaxisはフラグ的な意味になり、0は列、1は行と説明されるが、実際は「配列の次元」を指す。
			//　つまり対象の配列が２次元（縦横の行列）であれば０と１が意味を持ち、２は範囲外となる。
			//　還元すると「axis=2」を指定する場合、対象とする配列が３次元であることが暗黙の前提となる。
			var pResultTensor = pResults.Outputs[m_pModel.OutputFeatures.ElementAt(0).Name] as TensorFloat;

			for (int i= 0; i < pResultTensor.Shape.Count; i ++)
			{
				Debug.WriteLine($"Shape[{i}]={pResultTensor.Shape[i]}");
			}

			// Pythonで学ぶ画像認識 機械学習実践シリーズ

			var pResultVector = pResultTensor.GetAsVectorView();

			//　[N, 21, Height, Width]
			//　→　FCNの出力は、21のレイヤーをピクセル毎に、それが何であるかを分類する。
			//　つまり、その画素が該当クラスにどの程度の確立でそれであるか？を示す。
			//　正直、このままだと応用が難しい。（2024/02/15）
			//　このため、実装を中断する。
			var nCount = pResultVector.Count / 4;
			var iResultVector = pResultVector.GetEnumerator();
			/*
			while (iResultVector.MoveNext() )
			{
				var  = iResultVector.Current;
			}
			for (int i = 0; i < nCount; i++)
			{

				;
			}
			*/


			var argmax = pResultVector.Select((x, i) => new { x, i }).Aggregate((max, xi) => xi.x > max.x ? xi : max).i;




			List<(int index, float probability)> indexedResults = new List<(int, float)>();
			for (int i = 0; i < pResultVector.Count; i++)
			{
				var Value = pResultVector.ElementAt(i);
				Debug.WriteLine($"Value={Value}");

				indexedResults.Add((index: i, probability: pResultVector.ElementAt(i)));
			}
			indexedResults.Sort((a, b) =>
			{
				if (a.probability < b.probability)
				{
					return 1;
				}
				else if (a.probability > b.probability)
				{
					return -1;
				}
				else
				{
					return 0;
				}
			});

			for (int i = 0; i < 3; i++)
			{
				//Console.WriteLine($"\"{_labels[indexedResults[i].index]}\" with confidence of {indexedResults[i].probability}");
			}
		}
	}
}
