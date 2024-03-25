//　YOLOv4.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/DenseNet.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
//　https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
//　https://github.com/NVIDIA/retinanet-examples/blob/main/extras/cppapi/README.md
//　https://github.com/AlexeyAB/darknet/wiki

//　パラメータか使い方に誤りがあるのか、判定結果のスレッシュホールドに斑がある。（2024/03/24）
CDenseNet::CDenseNet()
{
	//　フレームワークは、Caffe
	//　https://github.com/shicai/DenseNet-Caffe
	//　
	//　実装サンプル（c++）
	//　https://github.com/opencv/opencv/blob/8c25a8eb7b10fb50cda323ee6bec68aa1a9ce43c/samples/dnn/object_detection.cpp#L192-L221
	m_pModelFilepath   = ::GetAssetFolder();
	m_pModelFilepath  += "Networks\\Caffe\\DenseNet\\DenseNet_121.caffemodel";
	m_pConfigFilepath  = ::GetAssetFolder();
	m_pConfigFilepath += "Networks\\Caffe\\DenseNet\\DenseNet_121.prototxt";
	m_pLabelFilepath   = ::GetAssetFolder();
	m_pLabelFilepath  += "Networks\\Common\\classification_classes_ILSVRC2012.txt";		//　各行のカンマで区切られた最初の要素が名前ラベル
	m_pFrameWorkName   = "Caffe";

	//　Caffeの入力矩形は、prototxtに記述されている。（2024/03/24）
	m_fInputShape.width  = 224;
	m_fInputShape.height = 224;

	m_dThreshold = 0.92;
}

CDenseNet::~CDenseNet()
{
}

//　
cv::Mat
CDenseNet::Prepare(cv::Mat& pImage)
{
	auto pBlob = cv::dnn::blobFromImage(pImage, 0.017, m_fInputShape, cv::Scalar(103.94, 116.78, 123.68), true, false);
	return(pBlob);
}

//　
bool
CDenseNet::Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults)
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

//　
bool
CDenseNet::Dump(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults, std::vector<std::string> &  pNames)
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
