/*
@brief common library
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_HPP_
#define _COMMON_HPP_

#include <common/precomm.hpp>
#include <common/cmdline.hpp>
#include <common/codecvt.hpp>
#include <common/debuglog.hpp>
#include <common/windows.hpp>
#include <common/opencv.hpp>

namespace common {

    static const std::string _TAG = "common";

    /**
    *@brief 目录补全'\\'
    @return 若非目录返回原字符串
    */
    template<typename T>
    auto fillDir(const T* dir) {
        size_t n = dir.find_last_of('\\');
        if (n == static_cast<size_t>(-1)) {}
        else if (n != dir.size() - 1) { dir = dir + _T("\\"); }
        return std::basic_string<T, std::char_traits<T>, std::allocator<T>>(dir);
    }

} // namespace common

#endif // _COMMON_HPP_