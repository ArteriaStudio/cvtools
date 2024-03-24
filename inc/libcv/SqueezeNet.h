#pragma 	once
//　SqueezeNet.h
#ifndef 	__SQUEEZENET_H__
#include	"libcv/DnnBase.h"

//　他のモデルと比べて、かなり情報が豊富で複数のフレームワークへの転用実績がある気配（2024/03/22）
//　https://github.com/AlexeyAB/darknet?tab=readme-ov-file#yolo-v4-in-other-frameworks
class	CSqueezeNet : public CDnnBase
{
protected:

public:
	 CSqueezeNet();
	~CSqueezeNet();

	cv::Mat 	Prepare(cv::Mat &  pImage);
	bool		Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults);
};

#endif	//	__SQUEEZENET_H__
