/*
@brief pre common header.
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _PRECOMM_H_
#define _PRECOMM_H_
#include <cstdlib>

namespace common {

    ///----------资源初始化----------

    /**
    *@brief memset 0
    *@param p 类型指针
    *@parma length 长度
    */
    template<typename T>
    inline void zeroset(T& p, size_t length)
    {
        std::memset(p, 0, sizeof(*p)*length);
    }

    /**
    *@brief wmemset 0
    *@param p 类型指针
    *@parma length 长度
    */
    template<typename T>
    inline void wzeroset(T& p, size_t length)
    {
        std::wmemset(p, 0, sizeof(*p)*length);
    }

    ///----------资源安全释放----------

    /**
    *@brief free_s 可接受不定长参数
    */
    template<typename T>
    inline void free_s(T& p)
    {
        if (p != nullptr) { std::free(static_cast<void*>(p)); p = nullptr; }
    }

    /**
    *@brief free_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void free_s(T& p, Args&... args)
    {
        if (p != nullptr) { std::free(static_cast<void*>(p)); p = nullptr; }
        free_s(args...);
    }

    /**
    *@brief delete_s 可接受不定长参数
    */
    template<typename T>
    inline void delete_s(T& p)
    {
        if (p != nullptr) { delete(p); p = nullptr; }
    }

    /**
    *@brief delete_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void delete_s(T& p, Args&... args)
    {
        if (p != nullptr) { delete(p); p = nullptr; }
        delete_s(args...);
    }

    /**
    *@brief delete[]_s 可接受不定长参数
    */
    template<typename T>
    inline void deleteA_s(T& p)
    {
        if (p != nullptr) { delete[](p); p = nullptr; }
    }

    /**
    *@brief delete[]_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void deleteA_s(T& p, Args&... args)
    {
        if (p != nullptr) { delete[](p); p = nullptr; }
        deleteA_s(args...);
    }

    /**
    *@brief Release_s 可接受不定长参数
    */
    template<typename T>
    inline void Release_s(T& p)
    {
        if (p != nullptr) { p->Release(); p = nullptr; }
    }

    /**
    *@brief Release_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void Release_s(T& p, Args&... args)
    {
        if (p != nullptr) { p->Release(); p = nullptr; }
        Release_s(args...);
    }

    /**
    *@brief release_s 可接受不定长参数
    */
    template<typename T>
    inline void release_s(T& p)
    {
        if (p != nullptr) { p->release(); p = nullptr; }
    }

    /**
    *@brief release_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void release_s(T& p, Args&... args)
    {
        if (p != nullptr) { p->release(); p = nullptr; }
        release_s(args...);
    }

}
#endif // _PRECOMM_H_