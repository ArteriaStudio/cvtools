#include	"framework.h"
#include	"cv.h"



bool
TrackerVideo(const char * pImageFilepath)
{
	/*
	auto pSiamRPN_Params = cv::TrackerDaSiamRPN::Params();
	//pSiamRPN_Params.
	auto b = cv::TrackerDaSiamRPN::create(pSiamRPN_Params);
	*/

	//　DNN 動画用トラッカーは、内側で静止画用の物体検出モデルを使用している。（2024/01/25）
	auto c = cv::TrackerGOTURN::create();
	auto d = cv::TrackerMIL::create();
	auto e = cv::TrackerNano::create();

	auto pImage = cv::imread(pImageFilepath);
	cv::Rect2d	pBox;

	auto font_scale = 3;
	auto font = cv::FONT_HERSHEY_PLAIN;

	c->init(pImage, pBox);

	std::string	pText = std::format("{},{},{},{}", pBox.x, pBox.y, pBox.width, pBox.height);

	cv::rectangle(pImage, pBox, cv::Scalar(0, 255, 0), 3);
	cv::putText(pImage, pText, cv::Point((int)pBox.x + 10, (int)pBox.y + 40), font, font_scale, cv::Scalar(0, 0, 255), 3);

	cv::imshow("image", pImage);
	cv::waitKey();

	return(true);
}