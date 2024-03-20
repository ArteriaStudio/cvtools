//　RetinaNet.cpp
#include	"framework.h"
#include	"libcv/RetinaNet.h"


//　https://github.com/NVIDIA/retinanet-examples/blob/main/extras/cppapi/README.md

CRetinaNet::CRetinaNet()
{
	m_pModelFilepath = ::GetAssetFolder();
	m_pModelFilepath += "Networks\\RetinaNet\\retinanet-9.onnx";
}

CRetinaNet::~CRetinaNet()
{
}

