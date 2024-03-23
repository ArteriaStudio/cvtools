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



CDnnInfence::CDnnInfence()
{
	x = y = w = h = 0;
	iClassId = INT_MAX;
	fConfidence = 0.0f;
}

CDnnInfence::~CDnnInfence()
{
}

CDnnNetBase::CDnnNetBase()
{
	;
}

CDnnNetBase::~CDnnNetBase()
{
	;
}

//　ネットワークモデルを生成
bool
CDnnNetBase::Create()
{
	m_pNetModel = cv::dnn::readNet(m_pModelFilepath, m_pConfigFilepath, m_pFrameWorkName);
	if (m_pNetModel.empty() == true) {
		return(EXIT_FAILURE);
	}
	m_pNetModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	m_pNetModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	return(true);
}

//　配布されたモデルをONNXに変換して使用するのは互換性の問題を孕む（2024/03/20）
cv::Mat
CDnnNetBase::Execute(cv::Mat & pBlob)
{
	std::vector<cv::String> 	pOutNames = m_pNetModel.getUnconnectedOutLayersNames();
#ifdef		_DEBUG
	::DumpStrings(pOutNames);
#endif	//	_DEBUG

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


//　TensorFlow 2 Model
//　https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

bool
CDnnNetBase::Post(cv::Mat &  pImage, std::vector<cv::Mat> &  pOuts, VDnnInfences &  pResults)
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
				int width = (int)(data[2] * pImage.cols);
				int height = (int)(data[3] * pImage.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

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

	return(true);
}
