//　libcv.cpp
#include	"framework.h"
#include	"libcv/libcv.h"




CDnnBase::CDnnBase()
{
}

CDnnBase::~CDnnBase()
{
}

//　TensorFlow 2 Model
//　https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

//　配布されたモデルをONNXに変換して使用するのは互換性の問題を孕む（2024/03/20）
bool
CDnnBase::Execute()
{
//	auto pModelFilepath = "D:\\Home\\rink\\projects\\assets\\networks\\RetinaNet\\retinanet-9.onnx";
	auto pConfigFilepath = "";
	//	auto pModelFilepath  = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\frozen_inference_graph.pb";
	//	auto pConfigFilepath = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
	auto pNetModel = cv::dnn::readNetFromONNX(m_pModelFilepath.c_str());
	//	auto pNetModel = cv::dnn::readNet(pModelFilepath);
	if (pNetModel.empty() == true) {
		return(EXIT_FAILURE);
	}
	pNetModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	pNetModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

//	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\_a5807cd9-ee60-4366-b593-6db18aba2ef1.jpg";
//	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\search.png";
	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\img_c8dad835c4b9134b067cc8b8efcab22f143142.jpg";
	auto pImage = cv::imread(pImageFilepath);
	auto nRows = pImage.rows;
	auto nCols = pImage.cols;
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0, cv::Size(nRows, nCols), cv::Scalar(0.485, 0.456, 0.406), true, false);

	pNetModel.setInput(pBlob);
	auto pOutStream = getOutputsNames(pNetModel);
//	auto pOut = pNetModel.forward(pOutStream[0]);
	try {
		auto pOut = pNetModel.forward();
		auto nChannels = pOut.channels();	//　成分数
		auto nDimensions = pOut.dims;		//　次元数
		/*
		pOut;
		auto b = pOut.ptr(0, 0);

		cv::Mat a;
		*/
		cv::imshow("image", pImage);
		cv::waitKey();
	} catch (...) {
		return(false);
	}

	return(true);
}




//　TensorFlowのモデルを他フレームワークのファイル形式に変換
//　https://github.com/onnx/tensorflow-onnx


FILE *	pOutFile = nullptr;
FILE *	pErrFile = nullptr;

bool
CvInitialize()
{
/*
	freopen_s(&pOutFile, "D:\\Tmp\\OpenCV_out.log", "a+", stdout);
	freopen_s(&pErrFile, "D:\\Tmp\\OpenCV_err.log", "a+", stderr);

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
	auto log_level = cv::utils::logging::getLogLevel();
*/
	return(true);
}

void
CvFinalize()
{
/*
	::fclose(pErrFile);
	::fclose(pOutFile);
*/
	return;
}

const char *
GetAssetFolder(void)
{
static std::string	g_pAssetFolder = "D:\\Home\\Rink\\projects\\assets\\";

	return(g_pAssetFolder.c_str());
}

bool
LoadStrings(std::vector<std::string> &  pList, const char *  pFilepath)
{
	FILE *	pFile = nullptr;

	if (fopen_s(&pFile, pFilepath, "r")) {
		return(false);
	}
	char	pText[256];
	while (fgets(pText, sizeof(pText), pFile) != nullptr) {
		auto nText = strlen(pText);
		if (pText[nText - 1] == '\n') {
			pText[nText - 1] = '\0';
		}
		pList.push_back(pText);
	}
	fclose(pFile);
	
	return(true);
}




// Get Output Layers Name
std::vector<std::string>	getOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<std::string> names;
	if (names.empty()) {
		std::vector<int32_t> out_layers = net.getUnconnectedOutLayers();
		std::vector<std::string> layers_names = net.getLayerNames();
		names.resize(out_layers.size());
		for (size_t i = 0; i < out_layers.size(); ++i) {
			names[i] = layers_names[out_layers[i] - 1];
		}
	}
	return names;
}

