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
        std::basic_string<T, std::char_traits<T>, std::allocator<T>> _dir = dir;
        size_t n = _dir.find_last_of(_T("\\"));
        if (n == static_cast<size_t>(-1)) {}
        else if (n != _dir.size() - 1) { _dir += _T("\\"); }
        return _dir;
    }

} // namespace common

#endif // _COMMON_HPP_