#pragma 	once
//�@libcv.h
#ifndef 	__MOBILENET_H__
#include	"libcv/DnnBase.h"

//�@
class	CMobileNet : public CDnnBase
{
protected:
public:
	 CMobileNet();
	~CMobileNet();

	cv::Mat 	Prepare(cv::Mat & pImage);
	cv::Mat 	Execute(cv::Mat & pBlob);
//	bool		Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults);
};

#endif	//	__MOBILENET_H__
