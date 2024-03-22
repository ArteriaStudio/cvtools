#pragma 	once
//�@MobileNetV3.h
//�@���̌��o���f���iSSD-MobileNetV3�j
#ifndef 	__MOBILENETV3_H__
#include	"libcv/DnnBase.h"

//�@
class	CMobileNetV3 : public CDnnNetBase
{
protected:
public:
	 CMobileNetV3();
	~CMobileNetV3();

	cv::Mat 	Prepare(cv::Mat & pImage);
	cv::Mat 	Execute(cv::Mat & pBlob);
};

#endif	//	__MOBILENETV3_H__
