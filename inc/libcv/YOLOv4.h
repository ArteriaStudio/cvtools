#pragma 	once
//　YOLOv4.h
#ifndef 	__YOLOV4_H__
#include	"libcv/DnnBase.h"

//　他のモデルと比べて、かなり情報が豊富で複数のフレームワークへの転用実績がある気配（2024/03/22）
//　https://github.com/AlexeyAB/darknet?tab=readme-ov-file#yolo-v4-in-other-frameworks
class	CYOLOv4 : public CDnnDetectBase
{
protected:

public:
	 CYOLOv4();
	~CYOLOv4();

	bool		Create();
	cv::Mat 	Prepare(cv::Mat &  pImage);
	cv::Mat 	Execute(cv::Mat &  pBlob);
	std::vector<cv::Mat>	ExecuteEx(cv::Mat &  pBlob);

	bool		Post(cv::Mat &  pImage, std::vector<cv::Mat> &  pOuts, VDnnInfences &  pResults);
};

#endif	//	__YOLOV4_H__
