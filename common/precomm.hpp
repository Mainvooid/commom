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
#include <limits>
#include <chrono>
/**
  @addtogroup common
  @{
    @defgroup detail detail
  @}
*/
namespace common {
    /// @addtogroup common
    /// @{
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
    /**@overload*/
    template< typename T = std::chrono::milliseconds, typename R, typename ...Args>
    auto getFnDuration(R(*func)(Args...)) {
        std::function<R(Args...)> Fn = func;
        return[=](Args...args)->auto {
            return getFnDuration(Fn, args...);
        };
    }

    //----------资源初始化----------

    /**
    *@brief memset 0
    */
    template<typename T>
    inline void zeroset(T& p, size_t length)
    {
        std::memset(p, 0, sizeof(*p) * length);
    }
    /**@overload*/
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
    /**@overload*/
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

    /**
    *@brief Release_s 可接受不定长参数
    */
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

    /**
    @brief 模板条件参数推断及条件函数调用,根据宽窄字符类型返回不同对象.
    */
    template<typename T, typename TA, typename TW>
    typename std::enable_if_t<std::is_same_v<T, char> || std::is_same_v<T, std::string>, TA> tvalue(TA a, TW) { return a; };
    /**@overload*/
    template<typename T, typename TA, typename TW>
    typename std::enable_if_t<std::is_same_v<T, wchar_t> || std::is_same_v<T, std::wstring>, TW> tvalue(TA, TW w) { return w; };

    /**
    @brief 返回字符串长度
    */
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
    /**@overload*/
    template<typename R, typename ...FArgs>
    inline std::function<R(FArgs...)> getFunction(R(*Fn)(FArgs...)) { return Fn; }//*@note _WIN64下__stdcall的调用约定会被隐式转为__cdecl(缺省)
    /**@overload*/
    template<typename R, typename ...FArgs>
    inline std::function<R(FArgs..., va_list)> getFunction(R(*Fn)(FArgs..., ...)) { return Fn; }

#ifndef _WIN64
    /**@overload*/
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
        typename std::conditional_t<
            std::is_same_v<TI, std::string> || std::is_same_v<TI, char>,
            std::istringstream, std::wistringstream> iss(arg);
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