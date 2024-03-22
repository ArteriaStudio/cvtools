using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Windows.AI.MachineLearning;
using Windows.Graphics.Imaging;
using Windows.System;

namespace cvclass
{
	public class SSDMobileNet : CVBase
	{
		protected string m_pModeFilepath = null;

		public SSDMobileNet(){
			m_pModeFilepath = "D:\\Home\\Datas\\Network\\SSD-MobilenetV1\\ssd_mobilenet_v1_13-qdq.onnx";
		}

		protected override void DumpResults(LearningModelEvaluationResult pResults)
		{
			var pResultTensor_Boxes = pResults.Outputs[m_pModel.OutputFeatures.ElementAt(0).Name] as TensorFloat;
			var pResultTensor_Label = pResults.Outputs[m_pModel.OutputFeatures.ElementAt(1).Name] as TensorInt64Bit;
			var pResultTensor_Score = pResults.Outputs[m_pModel.OutputFeatures.ElementAt(2).Name] as TensorFloat;

			var pResultVector_Boxes = pResultTensor_Boxes.GetAsVectorView();
			var pResultVector_Label = pResultTensor_Label.GetAsVectorView();
			var pResultVector_Score = pResultTensor_Score.GetAsVectorView();

			var nElements_Boxes = pResultTensor_Boxes.Shape[1];
			//var nElements_Score = pResultTensor_Score.Shape[1];

			var pSegments = new List<Segment>();
			for (var iElement = 0; iElement < nElements_Boxes; iElement ++)
			{
				var iBoxes = iElement * 4;

				var pSegment = new Segment();
				pSegment.xMin = pResultVector_Boxes[iBoxes + 0];
				pSegment.yMin = pResultVector_Boxes[iBoxes + 1];
				pSegment.xMax = pResultVector_Boxes[iBoxes + 2];
				pSegment.yMax = pResultVector_Boxes[iBoxes + 3];
				pSegment.iLabel = pResultVector_Label[iElement];
				pSegment.fScore = pResultVector_Score[iElement];

				pSegments.Add(pSegment);
			}

			pSegments.Sort((a, b) => {
				if (a.fScore < b.fScore) { return (-1); }
				if (b.fScore > a.fScore) { return (+1); }
				return (0);
			});

			for (int i = 0; i < 10; i ++)
			{
				Debug.WriteLine($"Label={pSegments[i].iLabel}, Score={pSegments[i].fScore}, Boxes={pSegments[i].xMin}, {pSegments[i].yMin}, {pSegments[i].xMax}, {pSegments[i].yMax}");
			}

			return;
		}
	}
}
