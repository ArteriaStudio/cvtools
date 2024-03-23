//　YOLOv4.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/YOLOv4.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
//　https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
//　https://github.com/NVIDIA/retinanet-examples/blob/main/extras/cppapi/README.md
//　https://github.com/AlexeyAB/darknet/wiki

CYOLOv4::CYOLOv4()
{
	//　フレームワークは、DarkNet
	//　https://github.com/AlexeyAB/darknet
	//　https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo
	//　DarkNet全般
	//　https://github.com/AlexeyAB/darknet/wiki
	//　実装サンプル（c++）
	//　https://github.com/opencv/opencv/blob/8c25a8eb7b10fb50cda323ee6bec68aa1a9ce43c/samples/dnn/object_detection.cpp#L192-L221
	m_pModelFilepath   = ::GetAssetFolder();
	m_pModelFilepath  += "Networks\\DarkNet\\YOLOv4\\yolov4.weights";
	m_pConfigFilepath  = ::GetAssetFolder();
	m_pConfigFilepath += "Networks\\DarkNet\\YOLOv4\\yolov4.cfg";
	m_pFrameWorkName   = "Darknet";

	//　Darknetの入力矩形は、cfgに記述されている。（2024/03/22）
	m_fInputShape.width  = 608;
	m_fInputShape.height = 608;
}

CYOLOv4::~CYOLOv4()
{
}

//　物体検出モデルを生成
bool
CYOLOv4::Create()
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
CYOLOv4::Prepare(cv::Mat& pImage)
{
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0 / 255.0, m_fInputShape, cv::Scalar(0.0, 0.0, 0.0), true, false);
	return(pBlob);
}

//　
cv::Mat
CYOLOv4::Execute(cv::Mat &  pBlob)
{
	std::vector<cv::String> 	pOutNames = m_pNetModel.getUnconnectedOutLayersNames();
#ifdef		_DEBUG
	::DumpStrings(pOutNames);
#endif	//	_DEBUG

	m_pNetModel.setInput(pBlob);
	auto pOutStream = getOutputsNames(m_pNetModel);
	try {
		std::vector<cv::Mat>	pOuts;
//		auto pOut = m_pNetModel.forward(pOuts);
		m_pNetModel.forward(pOuts, pOutNames);
		auto nChannels = pOuts[0].channels();	//　成分数
		auto nDimensions = pOuts[0].dims;		//　次元数
		return(pOuts[2]);
	} catch (...) {
		;
	}

	return(cv::Mat());
}

//　
bool
CYOLOv4::Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults)
{
	// Network produces output blob with a shape NxC where N is a number of
	// detected objects and C is a number of classes + 4 where the first 4
	// numbers are [center_x, center_y, width, height]
	float * 	data = (float*)pOut.data;
	for (int j = 0; j < pOut.rows; ++j, data += pOut.cols) {
		cv::Mat 	scores = pOut.row(j).colRange(5, pOut.cols);
		cv::Point	classIdPoint;
		double		confidence;
		minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
		if (confidence > 0.93) {
			int centerX = (int)(data[0] * pImage.cols);
			int centerY = (int)(data[1] * pImage.rows);
			int width   = (int)(data[2] * pImage.cols);
			int height  = (int)(data[3] * pImage.rows);
			int left    = centerX - width / 2;
			int top     = centerY - height / 2;

			CDnnInfence		pResult;
			pResult.x = left;
			pResult.y = top;
			pResult.w = width;
			pResult.h = height;
			pResult.iClassId = classIdPoint.x;
			pResult.fConfidence = (float)confidence;

			pResults.push_back(pResult);
		}
	}

	return(true);
}
