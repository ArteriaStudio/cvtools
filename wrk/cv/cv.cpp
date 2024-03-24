// cv.cpp : アプリケーションのエントリ ポイントを定義します。
//

#include	"framework.h"
#include	"cv.h"
#include	"libcv/libcv.h"
#include	"libcv/RetinaNet.h"
#include	"libcv/MobileNet.h"
#include	"libcv/MobileNetV2.h"
#include	"libcv/MobileNetV3.h"
#include	"libcv/YOLOV4.h"
#include	"libcv/DenseNet.h"
#include	"libcv/SqueezeNet.h"


#ifndef 	_DEBUG
#pragma 	comment(lib, "opencv_world490.lib")
#else
#pragma 	comment(lib, "opencv_world490d.lib")
#endif	//	_DEBUG

#pragma 	comment(lib, "Shlwapi.lib")

// グローバル変数:
HINSTANCE	hInst;

/*
名前
公式サイト

labelImg
https://github.com/tzutalin/labelImg
https://github.com/HumanSignal/labelImg/blob/master/readme/README.jp.rst
https://tech.aru-zakki.com/how-to-use-labelimg/

Cloud Annotations
https://cloud.annotations.ai

VoTT
https://github.com/microsoft/VoTT

LabelBox
https://github.com/Labelbox/Labelbox

other Annotation Tools
https://en.wikipedia.org/wiki/List_of_manual_image_annotation_tools

Label Studio
https://tech.aru-zakki.com/how-to-use-label-studio-for-object-detection/

*/


//　学習モデル作成
//　https://qiita.com/kosuke1701/items/0fc039147962880a8756
//　https://toa-hakobune.hatenablog.com/entry/2021/01/11/212459
//　https://dev-partner.i-pro.com/space/TPFAQ/1007060562/%E3%82%A2%E3%83%8E%E3%83%86%E3%83%BC%E3%82%B7%E3%83%A7%E3%83%B3%E3%83%84%E3%83%BC%E3%83%AB%E3%80%8ElabelImg%E3%80%8F%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%9FAI%E3%83%A2%E3%83%87%E3%83%AB%E4%BD%9C%E6%88%90
//　[EfficientDet]
//　https://github.com/google/automl/tree/master/efficientdet
//　https://endaaman.me/tips/training-effcientdet-pytorch
//　https://docs.nvidia.com/tao/tao-toolkit-archive/tao-30-2202/text/object_detection/efficientdet.html
//　https://colab.research.google.com/github/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/tutorial/train_shape.ipynb
//　[AutoML]（名称変更：AutoML Vision → VertexAI AutoML）
//　https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform?hl=ja
//　https://qiita.com/kccs_nobuaki-sakuragi/items/01cf72e38e0eb096557d
//　[顔検出]
//　https://qiita.com/UnaNancyOwen/items/f3db189760037ec680f3
//　[転移学習]
//　https://qiita.com/IchiLab/items/fd99bcd92670607f8f9b
//　https://github.com/Kazuhito00/NARUTO-HandSignDetection?tab=readme-ov-file
//　[OpenCV Build]
//　https://swallow-incubate.com/archives/blog/20200508/

//　
int APIENTRY
wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	if (::CvInitialize() == false) {
		return(EXIT_FAILURE);
	}

	std::vector<std::string>	pClassNames;

//	auto pNet = new CYOLOv4();
//	auto pNet = new CMobileNetV2();
	auto pNet = new CMobileNetV3();
	if (pNet->Create(pClassNames) == false) {
		return(EXIT_FAILURE);
	}

	std::vector<std::string>	pClassNames2;

	auto pNetClass = new CDenseNet();
//	auto pNetClass = new CSqueezeNet();
	if (pNetClass->Create(pClassNames2) == false) {
		return(EXIT_FAILURE);
	}




	std::vector<std::string>	pImageFilepaths;
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\img_c8dad835c4b9134b067cc8b8efcab22f143142.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\00773-2858724274.png");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\EuweACdWYAEukxb.jpeg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\human1.webp");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\search.png");
	pImageFilepaths.push_back("D:\\Tmp\\POV your waifu sits in front of you.png");
	pImageFilepaths.push_back("D:\\Tmp\\BracingEvoMi.png");
	pImageFilepaths.push_back("D:\\Tmp\\00547-00547-00852-3251821908.png");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\jewMI8n.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\0huxbDX.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\_a5807cd9-ee60-4366-b593-6db18aba2ef1.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\_4ac10f69-818b-4351-8127-38e540070a0b.jpg");

	auto iImage = 0;
	assert(iImage < pImageFilepaths.size());

	auto pImage = cv::imread(pImageFilepaths[iImage]);
	auto pBlob = pNet->Prepare(pImage);
	auto pOut  = pNet->Execute(pBlob);

	VDnnInfences	pResults;

	pNet->Post(pImage, pOut, pResults);
	pNet->Dump(pImage, pOut, pResults,  pClassNames);

	delete pNet;


	for (auto i = pResults.begin(); i != pResults.end(); i++) {
		//　画像の切り出し
		cv::Mat		pImage2;
		pImage2 = cv::Mat(pImage, cv::Rect(i->x, i->y, i->w, i->h));

		VDnnInfences	pResults2;

		auto pBlob2 = pNetClass->Prepare(pImage2);
		auto pOut2 = pNetClass->Execute(pBlob2);
		pNetClass->Post(pImage2, pOut2, pResults2);
		pNetClass->Dump(pImage2, pOut2, pResults2, pClassNames2);

		rectangle(pImage, cv::Point(i->x, i->y), cv::Point(i->x + i->w, i->y + i->h), cv::Scalar(255, 255, 255), 2);

		if (pResults2.size() > 0) {
			auto iClassId    = pResults2[0].iClassId;
			auto fConfidence = pResults2[0].fConfidence;
			std::string		pText;
			pText = std::format("{}:{}", pClassNames2[iClassId].c_str(), fConfidence);
//			pText = std::format("{}:{}", pClassNames[i->iClassId].c_str(), i->fConfidence);
			putText(pImage, pText.c_str(), cv::Point(i->x, i->y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 200), 2);
		}
	}

	delete pNetClass;

	cv::imshow("image", pImage);
//	cv::imshow("image", pImage2);
	cv::waitKey();


	::CvFinalize();

	return(EXIT_SUCCESS);
}
