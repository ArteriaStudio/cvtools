#pragma 	once
//　YOLOv4.h
#ifndef 	__DENSENET_H__
#include	"libcv/DnnBase.h"

//　他のモデルと比べて、かなり情報が豊富で複数のフレームワークへの転用実績がある気配（2024/03/22）
//　https://github.com/AlexeyAB/darknet?tab=readme-ov-file#yolo-v4-in-other-frameworks
class	CDenseNet : public CDnnBase
{
protected:

public:
	 CDenseNet();
	~CDenseNet();

	cv::Mat 	Prepare(cv::Mat &  pImage);
//	cv::Mat 	Execute(cv::Mat &  pBlob);
	bool		Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults);
};

#endif	//	__DENSENET_H__
