/*
@brief common library
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_HPP_
#define _COMMON_HPP_

//默认开启所有库支持
#ifndef HAVE_CUDA
#define HAVE_CUDA
#endif
#ifndef HAVE_OPENCL
#define HAVE_OPENCL
#endif
#ifndef HAVE_OPENCV
#define HAVE_OPENCV
#endif
#ifndef HAVE_DIRECTX
#define HAVE_DIRECTX
#endif

#include <common/precomm.hpp>
#include <common/cmdline.hpp>
#include <common/codecvt.hpp>
#include <common/debuglog.hpp>
#include <common/windows.hpp>
#include <common/opencv.hpp>
#include <common/cuda.hpp>

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