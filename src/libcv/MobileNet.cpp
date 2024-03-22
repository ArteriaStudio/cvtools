//　RetinaNet.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/MobileNet.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV

CMobileNet::CMobileNet()
{
	//　フレームワークは、TensorFlow（Frozenモデル）
	//　https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
	//　https://arxiv.org/pdf/1905.02244.pdf
	m_pModelFilepath   = ::GetAssetFolder();
	m_pModelFilepath  += "Networks\\TensorFlow\\MobileNet-SSDv3\\frozen_inference_graph.pb";
	m_pConfigFilepath  = ::GetAssetFolder();
	m_pConfigFilepath += "Networks\\TensorFlow\\MobileNet-SSDv3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
	m_pFrameWorkName   = "TensorFlow";

	//　TensorFlowの入力矩形は、pbtxtに記述されている。（2024/03/32）
	m_fInputShape.width  = 320;
	m_fInputShape.height = 320;
}

CMobileNet::~CMobileNet()
{
}

//　
cv::Mat
CMobileNet::Prepare(cv::Mat &  pImage)
{
	/*
	if (CDnnNetBase::Prepare(pImage, pBlob) == false) {
		return(false);
	}
	*/
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false);

	return(pBlob);
}

//　
cv::Mat
CMobileNet::Execute(cv::Mat & pBlob)
{
	return(CDnnNetBase::Execute(pBlob));
}
