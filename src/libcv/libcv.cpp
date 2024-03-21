//　libcv.cpp
#include	"framework.h"
#include	"libcv/libcv.h"

//　TensorFlowのモデルを他フレームワークのファイル形式に変換
//　https://github.com/onnx/tensorflow-onnx


#ifdef		ENABLE_REDIRECT_STDOUT
FILE *	pOutFile = nullptr;
FILE *	pErrFile = nullptr;
#endif	//	ENABLE_REDIRECT_STDOUT

//　
bool
CvInitialize()
{
#ifdef		ENABLE_REDIRECT_STDOUT
	freopen_s(&pOutFile, "D:\\Tmp\\OpenCV_out.log", "a+", stdout);
	freopen_s(&pErrFile, "D:\\Tmp\\OpenCV_err.log", "a+", stderr);

	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_VERBOSE);
	auto log_level = cv::utils::logging::getLogLevel();
#endif	//	ENABLE_REDIRECT_STDOUT

	return(true);
}

//　
void
CvFinalize()
{
#ifdef		ENABLE_REDIRECT_STDOUT
	::fclose(pErrFile);
	::fclose(pOutFile);
#endif	//	ENABLE_REDIRECT_STDOUT

	return;
}

//　
const char *
GetAssetFolder(void)
{
static std::string	g_pAssetFolder = "D:\\Home\\Rink\\projects\\assets\\";

	return(g_pAssetFolder.c_str());
}

//　
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
