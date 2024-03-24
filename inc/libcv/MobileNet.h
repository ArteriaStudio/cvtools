#pragma 	once
//　libcv.h
#ifndef 	__MOBILENET_H__
#include	"libcv/DnnBase.h"

//　MobileNet 基底クラス
class	CMobileNet : public CDnnBase
{
protected:
public:
	 CMobileNet();
	~CMobileNet()=0;

	bool	Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults);
};

#endif	//	__MOBILENET_H__
