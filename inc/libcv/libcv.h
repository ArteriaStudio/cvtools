#pragma 	once
//　libcv.h
#ifndef 	__LIBCV_H__


//　関数プロトタイプ
bool	CvInitialize();
void	CvFinalize();

const char *	GetAssetFolder(void);
std::vector<std::string>	getOutputsNames(const cv::dnn::Net & net);
bool	LoadStrings(std::vector<std::string> &  pList, const char *  pFilepath);
void	DumpStrings(std::vector<cv::String> &  pLists);

#endif	//	__LIBCV_H__
