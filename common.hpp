﻿/*
@brief common library
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_HPP_
#define _COMMON_HPP_

//默认开启所有库支持
#if !defined(HAVE_CUDA) && defined(_WIN64)
// 基于CUDA 10.0
#define HAVE_CUDA 
#endif

#ifndef HAVE_OPENCL
#define HAVE_OPENCL 
#endif

#ifndef HAVE_OPENCV
//基于OpenCV 4.0 with contrib
#define HAVE_OPENCV 
#endif

#ifndef HAVE_DIRECTX
//基于Microsoft DirectX SDK (June 2010)
#define HAVE_DIRECTX 
#endif

//本项目cuda目录下的.cu文件添加到工程后可以开启本宏,宏详细说明见cuda/README.md
//#define HAVE_CUDA_KERNEL 

#include <common/precomm.hpp>
#include <common/cmdline.hpp>
#include <common/codecvt.hpp>
#include <common/debuglog.hpp>
#include <common/windows.hpp>
#include <common/opencv.hpp>
#include <common/cuda.hpp>
#include <common/opencl.hpp>

#include <chrono>
#include <functional>

namespace common {

    static const std::string _TAG = "common";

    /**
    *@brief 函数计时(默认std::chrono::milliseconds)
    *@param Fn 函数对象,可用匿名函数包装代码片段来计时
    *@param args 函数参数
    *@return 相应单位的时间计数
    */
    template< typename T = std::chrono::milliseconds, typename R, typename ...FArgs, typename ...Args>
    auto getFnDuration(std::function<R(FArgs...)> Fn, Args&... args) {
        auto start = std::chrono::system_clock::now();
        Fn(args...);
        auto end = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<T>(end - start);
        return static_cast<double>(duration.count());
    }

} // namespace common

#endif // _COMMON_HPP_