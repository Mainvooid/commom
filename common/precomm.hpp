/*
@brief pre common header.
@author guobao.v@gmail.com
*/
#ifndef _COMMON_PRECOMM_HPP_
#define _COMMON_PRECOMM_HPP_
#include <cstdlib>
#include <tchar.h>
#include <sstream>
#include <iostream>
#include <vector>
#include <functional>
#include <unordered_map>
#include <memory>
#include <map>

/**
  @addtogroup common
  @{
    @defgroup detail detail
  @}
*/
namespace common {
    /// @addtogroup common
    /// @{
    //----------资源初始化----------

    /**
    *@brief memset 0
    */
    template<typename T>
    inline void zeroset(T& p, size_t length)
    {
        std::memset(p, 0, sizeof(*p) * length);
    }
    template<unsigned N, typename T>
    inline void zeroset(T(&p)[N])
    {
        std::memset(p, 0, N);
    }

    /**
    *@brief wmemset 0
    */
    template<typename T>
    inline void wzeroset(T& p, size_t length)
    {
        std::wmemset(p, 0, sizeof(*p) * length);
    }
    template<unsigned N, typename T>
    inline void wzeroset(T(&p)[N])
    {
        std::wmemset(p, 0, N);
    }

    //----------资源安全释放----------

    /**
    *@brief free_s 可接受不定长参数
    */
    template<typename T>
    inline void free_s(T p)
    {
        if (p != nullptr) { std::free(static_cast<void*>(p)); p = nullptr; }
    }

    /**
    *@brief free_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void free_s(T p, Args ... args)
    {
        if (p != nullptr) { std::free(static_cast<void*>(p)); p = nullptr; }
        free_s(args...);
    }

    /**
    *@brief delete_s 可接受不定长参数
    */
    template<typename T>
    inline void delete_s(T p)
    {
        if (p != nullptr) { delete(p); p = nullptr; }
    }

    /**
    *@brief delete_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void delete_s(T p, Args ... args)
    {
        if (p != nullptr) { delete(p); p = nullptr; }
        delete_s(args...);
    }

    /**
    *@brief delete[]_s 可接受不定长参数
    */
    template<typename T>
    inline void deleteA_s(T p)
    {
        if (p != nullptr) { delete[](p); p = nullptr; }
    }

    /**
    *@brief delete[]_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void deleteA_s(T p, Args ... args)
    {
        if (p != nullptr) { delete[](p); p = nullptr; }
        deleteA_s(args...);
    }

    template<typename T>
    inline void Release_s(T p)
    {
        if (p != nullptr) { p->Release(); p = nullptr; }
    }

    /**
    *@brief Release_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void Release_s(T p, Args ... args)
    {
        if (p != nullptr) { p->Release(); p = nullptr; }
        Release_s(args...);
    }

    /**
    *@brief release_s 可接受不定长参数
    */
    template<typename T>
    inline void release_s(T p)
    {
        if (p != nullptr) { p->release(); p = nullptr; }
    }

    /**
    *@brief release_s 接受不定长参数
    */
    template<typename T, typename...Args>
    inline void release_s(T p, Args ... args)
    {
        if (p != nullptr) { p->release(); p = nullptr; }
        release_s(args...);
    }

    //----------模板类型辅助----------


    template<typename CS, typename TA, typename TW>
    struct ttype_t;

    template<typename TA, typename TW>
    struct ttype_t<char, TA, TW> { typedef TA type; };

    template<typename TA, typename TW>
    struct ttype_t<wchar_t, TA, TW> { typedef TW type; };

    template<typename TA, typename TW>
    struct ttype_t<std::string, TA, TW> { typedef TA type; };

    template<typename TA, typename TW>
    struct ttype_t<std::wstring, TA, TW> { typedef TW type; };

    //----------模板条件参数推断及条件函数调用----------

    template<typename TA, typename TW>
    inline typename ttype_t<char, TA, TW>::type tvalue(char*, TA a, TW) { return a; };

    template<typename TA, typename TW>
    inline typename ttype_t<wchar_t, TA, TW>::type tvalue(wchar_t*, TA, TW w) { return w; }

    template<typename TA, typename TW>
    inline typename ttype_t<std::string, TA, TW>::type tvalue(std::string*, TA a, TW) { return a; };

    template<typename TA, typename TW>
    inline typename ttype_t<std::wstring, TA, TW>::type tvalue(std::wstring*, TA, TW w) { return w; }

    template<typename CS, typename TA, typename TW>
    inline typename ttype_t<CS, TA, TW>::type tvalue(TA a, TW w)
    {
        return tvalue<TA, TW>(static_cast<CS*>(0), a, w);
    }

    template<typename T>
    inline size_t cslen(const T* str)
    {
        return tvalue<T>(strlen, wcslen)(str);
    }

    /**
    *@brief 获取std::function对象
    */
    template<typename R, typename ...FArgs>
    inline std::function<R(FArgs...)> getFunction(std::function<R(FArgs...)> Fn) { return Fn; }

    template<typename R, typename ...FArgs>
    inline std::function<R(FArgs...)> getFunction(R(*Fn)(FArgs...)) { return Fn; }//*@note _WIN64下__stdcall的调用约定会被隐式转为__cdecl(缺省)

    template<typename R, typename ...FArgs>
    inline std::function<R(FArgs..., va_list)> getFunction(R(*Fn)(FArgs..., ...)) { return Fn; }

#ifndef _WIN64
    template<typename R, typename ...FArgs>
    inline std::function<R(FArgs...)> getFunction(R(__stdcall*Fn)(FArgs...)) { return Fn; }
#endif 

    namespace detail {
        /// @addtogroup detail
        /// @{
        /**
        @brief 函数入参及结果缓存，缓存入参和函数的执行结果，若入参存在则从缓存返回结果
        */
        template <typename R, typename... Args>
        std::function<R(Args...)> cache_fn(R(*func)(Args...))
        {
            auto result_map = std::make_shared<std::map<std::tuple<Args...>, R>>();
            return ([=](Args... args) {//延迟执行
                std::tuple<Args...> _args(args...);
                if (result_map->find(_args) == result_map->end()) {
                    (*result_map)[_args] = func(args...);//未找到相同入参，执行函数刷新缓存
                }
                return (*result_map)[_args];//返回缓存
            });
        }
        ///@}
    }// namespace detail

    /**
    @brief 函数对象缓存，若存在相同类型函数指针，则调用相应缓存函数获取缓存结果,可以大幅提高递归类函数的性能
    */
    template <typename R, typename...  Args>
    std::function<R(Args...)> cache_fn(R(*func)(Args...), bool flush = false)
    {
        using function_type = std::function<R(Args...)>;
        static std::unordered_map<decltype(func), function_type> functor_map;
        if (flush) {//明确要求刷新缓存
            return functor_map[func] = detail::cache_fn(func);
        }
        if (functor_map.find(func) == functor_map.end()) {
            functor_map[func] = detail::cache_fn(func);//未找到相同函数，执行函数刷新缓存
        }
        return functor_map[func];//返回缓存
    }

    //----------基于流的string/wstring与基本类型的互转----------

    /**
    @brief 基本数据类型转字符串
    */
    template<typename char_t, typename TI>
    inline auto convert_to_string(const TI& arg)
    {
        std::basic_stringstream<char_t, std::char_traits<char_t>, std::allocator<char_t>> str;
        str << arg;
        return str.str();
    }

    /**
    @brief 从字符串解析基本数据类型
    */
    template<typename TO, typename TI>
    inline auto convert_from_string(const TI& arg) noexcept(false)
    {
        TO ret;
        typename ttype_t<TI, std::istringstream, std::wistringstream>::type iss(arg);
        if (!(iss >> ret && iss.eof())) { throw std::bad_cast(); }
        return ret;
    }

    //----------其他----------

    /**
    *@brief 目录路径检查,统一分隔符且补全末尾分隔符
    */
    template<typename T>
    auto fillDir(const T* dir, const T* separator = tvalue<T>("\\", L"\\"))
    {
        std::basic_string<T, std::char_traits<T>, std::allocator<T>> _dir = dir;
        std::vector<const T*> separators = { tvalue<T>("\\",L"\\"), tvalue<T>("/",L"/") };
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
    /// @}
} // namespace common

#endif // _COMMON_PRECOMM_HPP_