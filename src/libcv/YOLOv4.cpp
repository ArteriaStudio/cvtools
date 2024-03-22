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

	/*
	auto pModelLayers = m_pNetModel.getLayerNames();
//	m_pNetModel.getOutputDetails();
	*/

	return(true);
}

//　
cv::Mat
CYOLOv4::Prepare(cv::Mat& pImage)
{
	/*
	m_fInputShape.width  = pImage.cols;
	m_fInputShape.height = pImage.rows;
	*/
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0 / 255.0, m_fInputShape, cv::Scalar(0.0, 0.0, 0.0), true, false);

	return(pBlob);
}

//　
std::vector<cv::Mat>
CYOLOv4::ExecuteEx(cv::Mat &  pBlob)
{
	/*
	auto pNet = cv::dnn::DetectionModel(m_pNetModel);
	pNet.setInputParams(1.0 / 255.0, m_fInputShape, cv::Scalar(0.0, 0.0, 0.0), true, false);
	std::vector<int>		pClassIds;
	std::vector<float>		pConfidences;
	std::vector<cv::Rect>	pBoxes;
	pNet.detect(pBlob, pClassIds, pConfidences, pBoxes);
	*/

	std::vector<cv::String> 	pOutNames = m_pNetModel.getUnconnectedOutLayersNames();
	DumpStrings(pOutNames);


	m_pNetModel.setInput(pBlob);
	auto pOutStream = getOutputsNames(m_pNetModel);
	try {
		std::vector<cv::Mat>	pOuts;
		//auto pOut = m_pNetModel.forward(pOuts);
		m_pNetModel.forward(pOuts, pOutNames);
		//auto nChannels = pOut.channels();	//　成分数
		//auto nDimensions = pOut.dims;		//　次元数
		return(pOuts);
	}
	catch (...) {
		;
	}

	return(cv::Mat());
}

//　
cv::Mat
CYOLOv4::Execute(cv::Mat &  pBlob)
{
	/*
	auto pNet = cv::dnn::DetectionModel(m_pNetModel);
	pNet.setInputParams(1.0 / 255.0, m_fInputShape, cv::Scalar(0.0, 0.0, 0.0), true, false);
	std::vector<int>		pClassIds;
	std::vector<float>		pConfidences;
	std::vector<cv::Rect>	pBoxes;
	pNet.detect(pBlob, pClassIds, pConfidences, pBoxes);
	*/

	std::vector<cv::String> 	pOutNames = m_pNetModel.getUnconnectedOutLayersNames();
	::DumpStrings(pOutNames);

	m_pNetModel.setInput(pBlob);
	auto pOutStream = getOutputsNames(m_pNetModel);
	try {
		std::vector<cv::Mat>	pOuts;
		//auto pOut = m_pNetModel.forward(pOuts);
		m_pNetModel.forward(pOuts, pOutNames);
		//auto nChannels = pOut.channels();	//　成分数
		//auto nDimensions = pOut.dims;		//　次元数
		return(pOuts[0]);
	} catch (...) {
		;
	}

	return(cv::Mat());
}

//　
bool
CYOLOv4::Post(cv::Mat &  pImage, std::vector<cv::Mat> &  pOuts, VDnnInfences &  pResults)
{
	std::vector<int>	outLayers = m_pNetModel.getUnconnectedOutLayers();
	std::string 		outLayerType_0 = m_pNetModel.getLayer(outLayers[0])->type;
	std::string 		outLayerType_1 = m_pNetModel.getLayer(outLayers[1])->type;
	std::string 		outLayerType_2 = m_pNetModel.getLayer(outLayers[2])->type;

	//　"Region"
	for (size_t i = 0; i < pOuts.size(); ++i) {
		// Network produces output blob with a shape NxC where N is a number of
		// detected objects and C is a number of classes + 4 where the first 4
		// numbers are [center_x, center_y, width, height]
		float* data = (float*)pOuts[i].data;
		for (int j = 0; j < pOuts[i].rows; ++j, data += pOuts[i].cols) {
			cv::Mat scores = pOuts[i].row(j).colRange(5, pOuts[i].cols);
			cv::Point classIdPoint;
			double confidence;
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
	}





/*
	//　画像の寸法を獲得
	auto nImageW = pImage.cols;
	auto nImageH = pImage.rows;
*/
/*
	auto x = pOut.at<float>(0, 1);
	auto y = pOut.at<float>(0, 2);
	auto w = pOut.at<float>(0, 3);
	auto h = pOut.at<float>(0, 4);
	auto c = pOut.at<float>(0, 5);
*/


/*
	//　
//	auto nRows = pOut.rows;
//	auto nCols = pOut.cols;
	auto nRows = pOut.rows;
	auto nCols = pOut.cols;
	cv::Mat 	pDetectionMat(nRows, nCols, CV_32F, pOut.ptr<float>());

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
*/

	return(true);
}
