//　RetinaNet.cpp
#include	"framework.h"
#include	"libcv/libcv.h"
#include	"libcv/RetinaNet.h"


//　https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV
//　https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
//　https://github.com/NVIDIA/retinanet-examples/blob/main/extras/cppapi/README.md

CRetinaNet::CRetinaNet()
{
	m_pModelFilepath = ::GetAssetFolder();
	m_pModelFilepath += "Networks\\RetinaNet\\retinanet-9.onnx";
}

CRetinaNet::~CRetinaNet()
{
}

