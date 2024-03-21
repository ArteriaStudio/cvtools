//　RetinaNet.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/MobileNetV2.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV

CMobileNetV2::CMobileNetV2()
{
	//　フレームワークは、TensorFlow（Frozenモデル）
	//　https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
	//　https://arxiv.org/pdf/1905.02244.pdf
	m_pModelFilepath   = ::GetAssetFolder();
	m_pModelFilepath  += "Networks\\TensorFlow\\MobileNet-SSDv2\\frozen_inference_graph.pb";
	m_pConfigFilepath  = ::GetAssetFolder();
	m_pConfigFilepath += "Networks\\TensorFlow\\MobileNet-SSDv2\\ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
	m_pFrameWorkName   = "TensorFlow";

	//　TensorFlowの入力矩形は、pbtxtに記述されている。（2024/03/32）
	m_fInputShape.width  = 300;
	m_fInputShape.height = 300;
}

CMobileNetV2::~CMobileNetV2()
{
}

//　
cv::Mat
CMobileNetV2::Prepare(cv::Mat &  pImage)
{
	/*
	m_fInputShape.width  = pImage.cols;
	m_fInputShape.height = pImage.rows;
	*/
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0, m_fInputShape, cv::Scalar(104, 117, 123), true, false);

	return(pBlob);
}

//　
cv::Mat
CMobileNetV2::Execute(cv::Mat & pBlob)
{
	return(CDnnBase::Execute(pBlob));
}
