//　DnnBase.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/DnnBase.h"




CDnnBase::CDnnBase()
{
}

CDnnBase::~CDnnBase()
{
}

//　ネットワークモデルを生成
bool
CDnnBase::Create()
{
	//　
	m_pNetModel = cv::dnn::readNet(m_pModelFilepath, m_pConfigFilepath, m_pFrameWorkName);
	if (m_pNetModel.empty() == true) {
		return(EXIT_FAILURE);
	}
	m_pNetModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	m_pNetModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	return(true);
}

//　
cv::Mat
CDnnBase::Prepare(cv::Mat & pImage)
{
	auto nRows = pImage.rows;
	auto nCols = pImage.cols;

	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false);
//	m_pNetModel.setInput(m_pBlob);

	return(pBlob);
}

//　TensorFlow 2 Model
//　https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

//　配布されたモデルをONNXに変換して使用するのは互換性の問題を孕む（2024/03/20）
cv::Mat
CDnnBase::Execute(cv::Mat & pBlob)
{
	m_pNetModel.setInput(pBlob);
	auto pOutStream = getOutputsNames(m_pNetModel);
//	auto pOut = pNetModel.forward(pOutStream[0]);
	try {
		auto pOut = m_pNetModel.forward();
		auto nChannels = pOut.channels();	//　成分数
		auto nDimensions = pOut.dims;		//　次元数
		return(pOut);
	} catch (...) {
		;
	}

	return(cv::Mat());
}

//　
bool
CDnnBase::Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults)
{
	cv::Mat 	pDetectionMat(pOut.size[2], pOut.size[3], CV_32F, pOut.ptr<float>());

	for (int i = 0; i < pDetectionMat.rows; i++) {
		int class_id = (int)pDetectionMat.at<float>(i, 1);
		float fConfidence = pDetectionMat.at<float>(i, 2);

		// Check if the detection is of good quality
		if (fConfidence > 0.45) {
			int box_x = static_cast<int>(pDetectionMat.at<float>(i, 3) * pImage.cols);
			int box_y = static_cast<int>(pDetectionMat.at<float>(i, 4) * pImage.rows);
			int box_width = static_cast<int>(pDetectionMat.at<float>(i, 5) * pImage.cols - box_x);
			int box_height = static_cast<int>(pDetectionMat.at<float>(i, 6) * pImage.rows - box_y);
			
			CDnnInfence		pResult;
			pResult.x = box_x;
			pResult.y = box_y;
			pResult.w = box_width;
			pResult.h = box_height;
			pResult.iClassId = class_id;
			pResult.fConfidence = fConfidence;

			pResults.push_back(pResult);
		}
	}

	return(true);
}


CDnnInfence::CDnnInfence()
{
	x = y = w = h = 0;
	iClassId = INT_MAX;
	fConfidence = 0.0f;
}

CDnnInfence::~CDnnInfence()
{
}

