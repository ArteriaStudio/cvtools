#pragma 	once
//�@libcv.h
#ifndef 	__MOBILENETV2_H__
#include	"libcv/DnnBase.h"

//�@
class	CMobileNetV2 : public CDnnBase
{
protected:
public:
	 CMobileNetV2();
	~CMobileNetV2();

	cv::Mat 	Prepare(cv::Mat & pImage);
	cv::Mat 	Execute(cv::Mat & pBlob);
};

#endif	//	__MOBILENETV2_H__
