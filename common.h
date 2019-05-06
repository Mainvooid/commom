﻿/*
@brief common library
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_H_
#define _COMMON_H_

#include <common/precomm.h>
#include <common/cmdline.h>
#include <common/codecvt.h>
#include <common/debuglog.h>
#include <iostream>

namespace common {
    const std::string _TAG = "common";

#if defined(_WIN32) && !defined(API)
#define API __declspec(dllexport)
#else
#define API __declspec(dllimport)
#endif

    template<class interfaceCls, class implCls>
    std::shared_ptr<interfaceCls> getClsPtr() noexcept
    {
        std::shared_ptr<interfaceCls> ptr;
        ptr.reset(new implCls);
        return ptr;
    }

}
#endif // _COMMON_H_