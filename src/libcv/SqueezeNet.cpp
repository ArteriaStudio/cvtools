//　SqueezeNet.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/SqueezeNet.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
//　https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
//　https://github.com/NVIDIA/retinanet-examples/blob/main/extras/cppapi/README.md
//　https://github.com/AlexeyAB/darknet/wiki

//　パラメータか使い方に誤りがあるのか、判定結果のスレッシュホールドに斑がある。（2024/03/24）
CSqueezeNet::CSqueezeNet()
{
	//　フレームワークは、DarkNet
	//　https://github.com/AlexeyAB/darknet
	//　https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo
	//　DarkNet全般
	//　https://github.com/AlexeyAB/darknet/wiki
	//　実装サンプル（c++）
	//　https://github.com/opencv/opencv/blob/8c25a8eb7b10fb50cda323ee6bec68aa1a9ce43c/samples/dnn/object_detection.cpp#L192-L221
	m_pModelFilepath   = ::GetAssetFolder();
	m_pModelFilepath  += "Networks\\Caffe\\SqueezeNet\\squeezenet_v1.1.caffemodel";
	m_pConfigFilepath  = ::GetAssetFolder();
	m_pConfigFilepath += "Networks\\Caffe\\SqueezeNet\\squeezenet_v1.1.prototxt";
	m_pLabelFilepath   = ::GetAssetFolder();
	m_pLabelFilepath  += "Networks\\Common\\classification_classes_ILSVRC2012.txt";		//　各行のカンマで区切られた最初の要素が名前ラベル
	m_pFrameWorkName   = "Caffe";

	//　Caffeの入力矩形は、prototxtに記述されている。（2024/03/24）
	m_fInputShape.width  = 227;
	m_fInputShape.height = 227;

	m_dThreshold = 0.192;
}

CSqueezeNet::~CSqueezeNet()
{
}

//　
cv::Mat
CSqueezeNet::Prepare(cv::Mat& pImage)
{
	auto pBlob = cv::dnn::blobFromImage(pImage, 0.017, m_fInputShape, cv::Scalar(104, 117, 123), true, false);
	return(pBlob);
}

//　
bool
CSqueezeNet::Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults)
{
	cv::Point	pClassIdPoint;
	double		dFinal_prob;
	minMaxLoc(pOut.reshape(1, 1), 0, &dFinal_prob, 0, &pClassIdPoint);
	int label_id = pClassIdPoint.x;

	CDnnInfence		pResult;
	pResult.x = 8;
	pResult.y = 8;
/*
	pResult.w = width;
	pResult.h = height;
*/
	pResult.iClassId = pClassIdPoint.x;
	pResult.fConfidence = (float)dFinal_prob;

	pResults.push_back(pResult);

	return(true);
}
