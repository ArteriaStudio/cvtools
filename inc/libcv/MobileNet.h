#pragma 	once
//�@libcv.h
#ifndef 	__MOBILENET_H__
#include	"libcv/DnnBase.h"

//�@
class	CMobileNet : public CDnnNetBase
{
protected:
public:
	 CMobileNet();
	~CMobileNet();

	cv::Mat 	Prepare(cv::Mat & pImage);
	cv::Mat 	Execute(cv::Mat & pBlob);
};

#endif	//	__MOBILENET_H__
