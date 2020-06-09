/*
@brief common library
@author guobao.v@gmail.com
@section License MIT
@see https://github.com/Mainvooid/common
*/
#ifndef _COMMON_HPP_
#define _COMMON_HPP_

/**
@def HAVE_OPENCL
@brief 基于OpenCL 1.2

@def HAVE_OPENCV
@brief 基于OpenCV 4.0 with contrib

@def HAVE_DIRECTX
@brief 基于Microsoft DirectX SDK (June 2010)

@def HAVE_CUDA
@brief 基于CUDA 10.0

@def HAVE_CUDA_KERNEL
@brief 本项目cuda目录下的.cu文件添加到工程后可以开启本宏
@see common\cuda\README.md
*/
//默认关闭库支持
//#define HAVE_OPENCL
//#define HAVE_OPENCV
//#define HAVE_DIRECTX
//#define HAVE_CUDA
//#define HAVE_CUDA_KERNEL
//#define LINK_LIB_OPENCV_WORLD
//#define WITH_OPENCV_CONTRIB 需要编译cv时增加扩展库

//其他开关
// #define PRINT_TO_CONSOLE

#include <common/precomm.hpp>
#include <common/cmdline.hpp>
#include <common/codecvt.hpp>
#ifdef _WIN32
#include <common/debuglog.hpp>
#include <common/windows.hpp>
#endif
#ifdef HAVE_OPENCV
#include <common/opencv.hpp>
#endif
#if defined(HAVE_CUDA) && defined(_WIN64)
#include <common/cuda.hpp>
#endif
#ifdef HAVE_OPENCL
#include <common/opencl.hpp>
#endif

/**
  @defgroup common common
*/
namespace common {
    /// @addtogroup common
    /// @{
    static const ::std::string _TAG = "common";
    /// @}
} // namespace common

#endif // _COMMON_HPP_
