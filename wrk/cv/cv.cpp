﻿// cv.cpp : アプリケーションのエントリ ポイントを定義します。
//

#include	"framework.h"
#include	"cv.h"

#pragma comment(lib, "opencv_world490d.lib")

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

int		Main0003();
int		Main0004();

int APIENTRY
wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR lpCmdLine, _In_ int nCmdShow)
{
	UNREFERENCED_PARAMETER(hPrevInstance);
	UNREFERENCED_PARAMETER(lpCmdLine);

	Main0004();


	return(0);
}

// Get Output Layers Name
std::vector<std::string> getOutputsNames(const cv::dnn::Net& net)
{
	static std::vector<std::string> names;
	if (names.empty()) {
		std::vector<int32_t> out_layers = net.getUnconnectedOutLayers();
		std::vector<std::string> layers_names = net.getLayerNames();
		names.resize(out_layers.size());
		for (size_t i = 0; i < out_layers.size(); ++i) {
			names[i] = layers_names[out_layers[i] - 1];
		}
	}
	return names;
}

//　https://www.koi.mashykom.com/opencv.html#fd04
//　https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/
//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV

int
Main0004()
{
	auto pModelFilepath  = "D:\\Home\\rink\\projects\\assets\\networks\\retinanet-9.onnx";
	auto pConfigFilepath = "";
//	auto pModelFilepath  = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\frozen_inference_graph.pb";
//	auto pConfigFilepath = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
	auto pNetModel = cv::dnn::readNetFromONNX(pModelFilepath);
//	auto pNetModel = cv::dnn::readNet(pModelFilepath);
	if (pNetModel.empty() == true) {
		return(EXIT_FAILURE);
	}
	pNetModel.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	pNetModel.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

	//	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\_a5807cd9-ee60-4366-b593-6db18aba2ef1.jpg";
	//	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\search.png";
	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\img_c8dad835c4b9134b067cc8b8efcab22f143142.jpg";
	auto pImage = cv::imread(pImageFilepath);
	auto nRows = pImage.rows;
	auto nCols = pImage.cols;
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0, cv::Size(nRows, nCols), cv::Scalar(), true, false);

	pNetModel.setInput(pBlob);
	auto pOutStream = getOutputsNames(pNetModel);
//	auto pOut = pNetModel.forward(pOutStream[0]);
	auto pOut = pNetModel.forward();

	auto nChannels = pOut.channels();	//　成分数
	auto nDimensions = pOut.dims;		//　次元数


	pOut;


	auto b = pOut.ptr(0, 0);

	cv::Mat a;

	cv::imshow("image", pImage);
	cv::waitKey();

	return(0);
}


int
TrackVideo()
{
	std::vector<std::string>	pImageFilepaths;
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\_a5807cd9-ee60-4366-b593-6db18aba2ef1.jpg");	//　←判定なし
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\search.png");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\img_c8dad835c4b9134b067cc8b8efcab22f143142.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\bIZrJGz.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\Insta_bae.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\images.png");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\large.jpg");	//　ひまわり
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\kite.webp");	//　凧
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\blender.webp");	//　ミキサー
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\blender2.jpg");	//　凧ミキサー（右９０度回転）
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\human1.webp");	//　人

	auto iIndex = 10;
	auto pImageFilepath = pImageFilepaths[iIndex];

	TrackerVideo(pImageFilepath.c_str());

	return(0);
}

int
Main0003()
{
	//　TensorFlow on OpenCV
	//　https://www.koi.mashykom.com/opencv.html#fd05

	//　Prepare TensorFlow Model for OpenCV use.
	// https://github.com/opencv/opencv/pull/17384
	// https://github.com/tensorflow/tensorflow/issues/30865
	// https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms
	//　[DirectMLで機械学習：PythonをDirectMLで動かすイメージ]
	//　https://learn.microsoft.com/ja-jp/windows/ai/windows-ml/tutorials/tensorflow-intro
	//　https://qiita.com/_matuzaki/items/8813ca3347c83fbeaade

	//　モデル入力は終了し、物体検出時に例外が発生
//	auto pModelFilepath  = "D:\\Tmp\\tensor\\MaskRCNN\\frozen_inference_graph.pb";
//	auto pConfigFilepath = "D:\\Tmp\\tensor\\MaskRCNN\\mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";

	//　認識なし
//	auto pModelFilepath  = "D:\\Tmp\\tensor\\MobileNet-SSD v1 PPN\\frozen_inference_graph.pb";
//	auto pConfigFilepath = "D:\\Tmp\\tensor\\MobileNet-SSD v1 PPN\\ssd_mobilenet_v1_ppn_coco.pbtxt";

	//　認識なし
//	auto pModelFilepath  = "D:\\Tmp\\tensor\\MobileNet-SSD v2\\frozen_inference_graph.pb";
//	auto pConfigFilepath = "D:\\Tmp\\tensor\\MobileNet-SSD v2\\ssd_mobilenet_v2_coco_2018_03_29.pbtxt";

	//　認識あり
//	auto pModelFilepath  = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\frozen_inference_graph.pb";
//	auto pConfigFilepath = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";

	//　モデルの入力中にスタックオーバーフロー
//	auto pModelFilepath  = "D:\\Tmp\\tensor\\efficientdet-d0\\efficientdet-d0.pb";
//	auto pConfigFilepath = "D:\\Tmp\\tensor\\efficientdet-d0\\efficientdet-d0.pbtxt";

	//　
//	auto pModelFilepath  = "D:\\Home\\rink\\projects\\assets\\networks\\fcn-resnet50-12-int8.onnx";
//	auto pConfigFilepath = "";

	//　
	auto pModelFilepath  = "D:\\Home\\rink\\projects\\assets\\networks\\retinanet-9.onnx";
	auto pConfigFilepath = "";

	//　Windows Machine Learningと異なり、色々な形式に対応している。
//	auto pNetModel = cv::dnn::SegmentationModel(pModelFilepath, pConfigFilepath);
//	auto pNetModel = cv::dnn::ClassificationModel(pModelFilepath, pConfigFilepath);
//	auto pNetModel = cv::dnn::DetectionModel(pModelFilepath);
	auto pNetModel = cv::dnn::DetectionModel(pModelFilepath, pConfigFilepath);

	pNetModel.setInputSize(320, 320);
	pNetModel.setInputScale(1.0 / 127.5);
	pNetModel.setInputMean(cv::Scalar(127.5, 127.5, 127.5));
	pNetModel.setInputSwapRB(true);

	std::vector<std::string>	pImageFilepaths;
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\_a5807cd9-ee60-4366-b593-6db18aba2ef1.jpg");	//　←判定なし
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\search.png");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\img_c8dad835c4b9134b067cc8b8efcab22f143142.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\bIZrJGz.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\Insta_bae.jpg");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\images.png");
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\large.jpg");	//　ひまわり
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\kite.webp");	//　凧
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\blender.webp");	//　ミキサー
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\blender2.jpg");	//　凧ミキサー（右９０度回転）
	pImageFilepaths.push_back("C:\\Users\\Rink\\OneDrive\\Pictures\\human1.webp");	//　人

	auto iIndex = 10;
	auto pImageFilepath = pImageFilepaths[iIndex];


	auto pImage = cv::imread(pImageFilepath);
	assert(*pImage.size.p);
	auto nRows = pImage.rows;
	auto nCols = pImage.cols;


	//　https://github.com/ChiekoN/OpenCV_SSD_MobileNet/blob/master/model/object_detection_classes_coco.txt
	std::vector<cv::String>	pClassLabels;

	pClassLabels.push_back("undefined");
	pClassLabels.push_back("person");
	pClassLabels.push_back("bicycle");
	pClassLabels.push_back("car");
	pClassLabels.push_back("motorcycle");
	pClassLabels.push_back("airplane");
	pClassLabels.push_back("bus");
	pClassLabels.push_back("train");
	pClassLabels.push_back("truck");
	pClassLabels.push_back("boat");
	pClassLabels.push_back("traffic light");
	pClassLabels.push_back("fire hydrant");
	pClassLabels.push_back("street sign");
	pClassLabels.push_back("stop sign");
	pClassLabels.push_back("parking meter");
	pClassLabels.push_back("bench");
	pClassLabels.push_back("bird");
	pClassLabels.push_back("cat");
	pClassLabels.push_back("dog");
	pClassLabels.push_back("horse");
	pClassLabels.push_back("sheep");
	pClassLabels.push_back("cow");
	pClassLabels.push_back("elephant");
	pClassLabels.push_back("bear");
	pClassLabels.push_back("zebra");
	pClassLabels.push_back("giraffe");
	pClassLabels.push_back("hat");
	pClassLabels.push_back("backpack");
	pClassLabels.push_back("umbrella");
	pClassLabels.push_back("shoe");
	pClassLabels.push_back("eye glasses");
	pClassLabels.push_back("handbag");
	pClassLabels.push_back("tie");
	pClassLabels.push_back("suitcase");
	pClassLabels.push_back("frisbee");
	pClassLabels.push_back("skis");
	pClassLabels.push_back("snowboard");
	pClassLabels.push_back("sports ball");
	pClassLabels.push_back("kite");
	pClassLabels.push_back("baseball bat");
	pClassLabels.push_back("baseball glove");
	pClassLabels.push_back("skateboard");
	pClassLabels.push_back("surfboard");
	pClassLabels.push_back("tennis racket");
	pClassLabels.push_back("bottle");
	pClassLabels.push_back("plate");
	pClassLabels.push_back("wine glass");
	pClassLabels.push_back("cup");
	pClassLabels.push_back("fork");
	pClassLabels.push_back("knife");
	pClassLabels.push_back("spoon");
	pClassLabels.push_back("bowl");
	pClassLabels.push_back("banana");
	pClassLabels.push_back("apple");
	pClassLabels.push_back("sandwich");
	pClassLabels.push_back("orange");
	pClassLabels.push_back("broccoli");
	pClassLabels.push_back("carrot");
	pClassLabels.push_back("hot dog");
	pClassLabels.push_back("pizza");
	pClassLabels.push_back("donut");
	pClassLabels.push_back("cake");
	pClassLabels.push_back("chair");
	pClassLabels.push_back("couch");
	pClassLabels.push_back("potted plant");
	pClassLabels.push_back("bed");
	pClassLabels.push_back("mirror");
	pClassLabels.push_back("dining table");
	pClassLabels.push_back("window");
	pClassLabels.push_back("desk");
	pClassLabels.push_back("toilet");
	pClassLabels.push_back("door");
	pClassLabels.push_back("tv");
	pClassLabels.push_back("laptop");
	pClassLabels.push_back("mouse");
	pClassLabels.push_back("remote");
	pClassLabels.push_back("keyboard");
	pClassLabels.push_back("cell phone");
	pClassLabels.push_back("microwave");
	pClassLabels.push_back("oven");
	pClassLabels.push_back("toaster");
	pClassLabels.push_back("sink");
	pClassLabels.push_back("refrigerator");
	pClassLabels.push_back("blender");
	pClassLabels.push_back("book");
	pClassLabels.push_back("clock");
	pClassLabels.push_back("vase");
	pClassLabels.push_back("scissors");
	pClassLabels.push_back("teddy bear");
	pClassLabels.push_back("hair drier");
	pClassLabels.push_back("toothbrush");



	//　
	std::vector<int>		pClassIds;
	std::vector<float>		pConfidences;
	std::vector<cv::Rect>	pBoxes;

	float	fConfThreshold = 0.45f;
	pNetModel.detect(pImage, pClassIds, pConfidences, pBoxes, fConfThreshold);
//	pNetModel.segment(pImage, pBoxes);

	//
	auto font_scale = 3;
	auto font = cv::FONT_HERSHEY_PLAIN;
	auto nElements = pClassIds.size();
	for (int iElement = 0; iElement < nElements; iElement++) {
		auto iClassId = pClassIds[iElement];
		auto iConfidence = pConfidences[iElement];
		auto pBox = pBoxes[iElement];
		std::string	pText = std::format("{}: {}", pClassLabels[iClassId], iConfidence);
		
		cv::rectangle(pImage, pBox, cv::Scalar(0, 255, 0), 3);
		cv::putText(pImage, pText, cv::Point(pBox.x + 10, pBox.y + 40), font, font_scale, cv::Scalar(0,0,255), 3);
	}



	cv::imshow("image", pImage);
	cv::waitKey();


	// モデルを凍結
	//　https://docs.openvino.ai/2021.4/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_EfficientDet_Models.html
	//　モデルに追加学習
	//　https://qiita.com/nekot0/items/3c0f8056811711641bbb
	//　https://github.com/xuannianz/EfficientDet
	//　https://learnopencv.com/object-tracking-using-opencv-cpp-python/

	return(0);
}

void
Main02()
{
	// BOOSTING MIL KCF TLD MEDIANFLOW GOTURN MOSSE CSRT
	return;
}

void
Main001()
{
	auto pModelFilepath  = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\frozen_inference_graph.pb";
	auto pConfigFilepath = "D:\\Tmp\\tensor\\MobileNet-SSD v3\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt";
	auto pNetModel = cv::dnn::readNetFromTensorflow(pModelFilepath, pConfigFilepath);

	//	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\_a5807cd9-ee60-4366-b593-6db18aba2ef1.jpg";
	//	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\search.png";
	auto pImageFilepath = "C:\\Users\\Rink\\OneDrive\\Pictures\\img_c8dad835c4b9134b067cc8b8efcab22f143142.jpg";
	auto pImage = cv::imread(pImageFilepath);
	auto nRows = pImage.rows;
	auto nCols = pImage.cols;
	auto pBlob = cv::dnn::blobFromImage(pImage, 1.0, cv::Size(300, 300), cv::Scalar(), true, false);

	pNetModel.setInput(pBlob);
	auto pOut = pNetModel.forward();

	auto nChannels = pOut.channels();	//　成分数
	auto nDimensions = pOut.dims;		//　次元数


	pOut;


	auto b = pOut.ptr(0, 0);

	cv::Mat a;

	cv::imshow("image", pImage);
	cv::waitKey();

	/*

	for detection in cvOut[0, 0, :, : ]:
	score = float(detection[2])
		if score > 0.3:
	left = detection[3] * cols
		top = detection[4] * rows
		right = detection[5] * cols
		bottom = detection[6] * rows
		cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness = 2)

		cv.imshow('img', img)
		cv.waitKey()
		;

	*/

}