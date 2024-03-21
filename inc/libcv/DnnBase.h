#pragma 	once
//　DnnBase.h
#ifndef 	__DNNBASE_H__

//　DNN オブジェクト分類結果
class	CDnnInfence
{
public:
	 CDnnInfence();
	~CDnnInfence();

	//　
	int 	x, y, w, h;		//　座標と範囲
	int 	iClassId;		//　判別識別子
	float	fConfidence;	//　確度
};
typedef std::vector<CDnnInfence>	VDnnInfences;


//　DNN 処理基底
class	CDnnBase
{
protected:
	std::string 	m_pModelFilepath;		//　ネットワークモデルファイル
	std::string 	m_pConfigFilepath;		//　ネットワーク構成情報ファイル
	std::string 	m_pFrameWorkName;		//　フレームワーク名
	cv::Size		m_fInputShape;			//　入力矩形の寸法

	cv::dnn::Net	m_pNetModel;	//　ネットワークモデル
//	cv::Mat 		m_pBlob;		//　入力テンソル

public:
			 CDnnBase();
	virtual ~CDnnBase()=0;

	virtual bool		Create();
	virtual cv::Mat 	Prepare(cv::Mat& pImage);
	virtual cv::Mat 	Execute(cv::Mat& pBlob);
	virtual bool		Post(cv::Mat &  pImage, cv::Mat &  pOut, VDnnInfences &  pResults);
};


//　関数プロトタイプ


#endif	//	__DNNBASE_H__
