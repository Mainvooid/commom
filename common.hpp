/*
@brief common library
@author guobao.v@gmail.com
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

@def HAVE_CUDA_DEVICE
@brief 本项目cuda目录下的.cu文件添加到工程后可以开启本宏
@see common\cuda\README.md
*/
//默认关闭库支持
#define HAVE_OPENCL
#define HAVE_OPENCV
#define HAVE_DIRECTX
#define HAVE_CUDA
#define HAVE_CUDA_DEVICE

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

#include <chrono>

/**
  @defgroup common common
*/
namespace common {
    /// @addtogroup common
    /// @{
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
    template< typename T = std::chrono::milliseconds, typename R, typename ...Args>
    auto getFnDuration(R(*func)(Args...)) {
        std::function<R(Args...)> Fn = func;
        return[=](Args...args)->auto {
            return getFnDuration(Fn, args...);
        };
    }

    /// @}
} // namespace common

#endif // _COMMON_HPP_