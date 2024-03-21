#pragma 	once
//Å@libcv.h
#ifndef 	__MOBILENETV3_H__
#include	"libcv/DnnBase.h"

//Å@
class	CMobileNetV3 : public CDnnBase
{
protected:
public:
	 CMobileNetV3();
	~CMobileNetV3();

	cv::Mat 	Prepare(cv::Mat & pImage);
	cv::Mat 	Execute(cv::Mat & pBlob);
};

#endif	//	__MOBILENETV3_H__
