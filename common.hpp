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
    *@brief 目录检查,分隔符统一且末尾分隔符补全
    */
    template<typename T>
    auto fillDir(const T* dir, const T* separator = _T("\\")) {
        std::basic_string<T, std::char_traits<T>, std::allocator<T>> _dir = dir;

        std::vector<const T*> separators = { _T("\\"), _T("/") };
        if (*separator == *separators[0]) {
            separators.erase(separators.begin());
        }
        size_t n = 0;
        while (true) {
            n = _dir.find_first_of(separators[0]);
            if (n == static_cast<size_t>(-1)) { break; }
            _dir.replace(n, 1, separator);
        }

        n = _dir.find_last_of(separator);
        if (n == static_cast<size_t>(-1) || n != _dir.size() - 1) { _dir += separator; }//无结尾分隔符
        return _dir;
    }


} // namespace common

#endif // _COMMON_HPP_