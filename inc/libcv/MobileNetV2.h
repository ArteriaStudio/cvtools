#pragma 	once
//�@MobileNetV2.h
//�@���̌��o���f���iSSD-MobileNetV2�j
#ifndef 	__MOBILENETV2_H__
#include	"libcv/MobileNet.h"

//�@
class	CMobileNetV2 : public CMobileNet
{
protected:
public:
	 CMobileNetV2();
	~CMobileNetV2();

	cv::Mat 	Prepare(cv::Mat &  pImage);
};

#endif	//	__MOBILENETV2_H__
