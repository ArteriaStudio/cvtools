//　MobileNetV3.cpp
//　物体検出モデル（SSD-MobileNetV3）
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/MobileNetV3.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV

CMobileNetV3::CMobileNetV3()
{
	//　フレームワークは、TensorFlow（Frozenモデル）
	//　https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
	//　https://arxiv.org/pdf/1905.02244.pdf
	m_pModelFilepath   = ::GetAssetFolder();
	m_pModelFilepath  += "Networks\\TensorFlow\\MobileNet-SSDv3\\frozen_inference_graph.pb";
	m_pConfigFilepath  = ::GetAssetFolder();
	m_pConfigFilepath += "Networks\\TensorFlow\\MobileNet-SSDv3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
	m_pLabelFilepath   = ::GetAssetFolder();
	m_pLabelFilepath  += "Networks\\TensorFlow\\object_detection_classes_coco.txt";
	m_pFrameWorkName   = "TensorFlow";

	//　TensorFlowの入力矩形は、pbtxtに記述されている。（2024/03/32）
	m_fInputShape.width  = 320;
	m_fInputShape.height = 320;

	m_dThreshold = 0.50;
//	m_dThreshold = 0.49;
}

CMobileNetV3::~CMobileNetV3()
{
}

//　
cv::Mat
CMobileNetV3::Prepare(cv::Mat &  pImage)
{
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0 / 127.5, cv::Size(320, 320), cv::Scalar(127.5, 127.5, 127.5), true, false);

	return(pBlob);
}
