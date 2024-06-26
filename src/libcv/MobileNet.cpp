//　RetinaNet.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/MobileNet.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
//　DL, CNN全般の解説
//　https://qiita.com/ba--shi/items/09f5f2f119ffbd9bb316

CMobileNet::CMobileNet()
{
}

CMobileNet::~CMobileNet()
{
}

//　PostProcessの参考実装
//　https://github.com/opencv/opencv/blob/8c25a8eb7b10fb50cda323ee6bec68aa1a9ce43c/samples/dnn/object_detection.cpp#L192-L221
bool
CMobileNet::Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults)
{
	//　
	auto nRows = pOut.size[2];
	auto nCols = pOut.size[3];
	cv::Mat 	pDetectionMat(nRows, nCols, CV_32F, pOut.ptr<float>());

	for (int i = 0; i < pDetectionMat.rows; i++) {
		int class_id = (int)pDetectionMat.at<float>(i, 1);
		float fConfidence = pDetectionMat.at<float>(i, 2);

		// Check if the detection is of good quality
		if (fConfidence > m_dThreshold) {
			int box_x = static_cast<int>(pDetectionMat.at<float>(i, 3) * pImage.cols);
			int box_y = static_cast<int>(pDetectionMat.at<float>(i, 4) * pImage.rows);
			int box_width = static_cast<int>(pDetectionMat.at<float>(i, 5) * pImage.cols - box_x);
			int box_height = static_cast<int>(pDetectionMat.at<float>(i, 6) * pImage.rows - box_y);
			
			CDnnInfence		pResult;
			pResult.x = box_x;
			pResult.y = box_y;
			pResult.w = box_width;
			pResult.h = box_height;
			pResult.iClassId = class_id - 1;	//　SSD-MobileNetは、1オリジンと思われる。（2024/03/22）
			pResult.fConfidence = fConfidence;

			pResults.push_back(pResult);
		}
	}

	return(true);
}

//　ダンプ
bool
CMobileNet::Dump(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults, std::vector<std::string> pNames)
{
	//　
	auto nRows = pOut.size[2];
	auto nCols = pOut.size[3];
	cv::Mat 	pDetectionMat(nRows, nCols, CV_32F, pOut.ptr<float>());

	for (int i = 0; i < pDetectionMat.rows; i++) {
		int class_id = (int)pDetectionMat.at<float>(i, 1);
		float fConfidence = pDetectionMat.at<float>(i, 2);

		std::string		pText;
		if (class_id < pNames.size()) {
			pText = std::format("{}:{}\n", pNames[class_id].c_str(), fConfidence);
		}
		else {
			pText = std::format("{}:{}\n", "(Unknown)", fConfidence);
		}
		::OutputDebugStringA(pText.c_str());
	}

	return(true);
}
