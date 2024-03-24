// header.h : 標準のシステム インクルード ファイルのインクルード ファイル、
// またはプロジェクト専用のインクルード ファイル
//

#pragma		once

#include	"targetver.h"
#define		WIN32_LEAN_AND_MEAN             // Windows ヘッダーからほとんど使用されていない部分を除外する

// Windows ヘッダー ファイル
#include	<windows.h>
#include	<shlwapi.h>

// C ランタイム ヘッダー ファイル
#include	<stdlib.h>
#include	<malloc.h>
#include	<memory.h>
#include	<tchar.h>
#include	<assert.h>

// C++ ランタイム ヘッダー ファイル
#include	<string>
#include	<format>

// OpenCV ヘッダーファイル
#include	<opencv2/opencv.hpp>
#include	<opencv2/dnn/dnn.hpp>
#include	<opencv2/video/tracking.hpp>
#include	<opencv2/core/utils/logger.hpp>

