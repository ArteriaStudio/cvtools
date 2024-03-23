#pragma 	once
//�@MobileNetV3.h
//�@���̌��o���f���iSSD-MobileNetV3�j
#ifndef 	__MOBILENETV3_H__
#include	"libcv/MobileNet.h"

//�@
class	CMobileNetV3 : public CMobileNet
{
protected:
public:
	 CMobileNetV3();
	~CMobileNetV3();

	cv::Mat 	Prepare(cv::Mat &  pImage);
};

#endif	//	__MOBILENETV3_H__
