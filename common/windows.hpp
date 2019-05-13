/*
@brief dll helper
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#if !defined(_COMMON_WINDOWS_HPP_) && defined(_WIN32)
#define _COMMON_WINDOWS_HPP_

#include <common/precomm.hpp>
#include <memory>
#include <iostream>
#include <windows.h>

#ifndef DLLAPI
#define DLLAPI __declspec(dllexport)
#else
#define DLLAPI __declspec(dllimport)
#endif

namespace common {

    namespace windows {

        ///DLL导出

        /**
        *@brief 用于dll导出接口类的实现类指针,自动管理内存,以防范Cross-Dll问题
        */
        template<class interfaceCls, class implCls>
        std::shared_ptr<interfaceCls> getClsPtr() noexcept
        {
            return std::make_shared<implCls>();
        }

        ///DLL导入

       /**
        *@brief 返回工作目录
        */
        template<typename T>
        auto getWorkDir() noexcept
        {
            T current_exe_fname[MAX_PATH];
            T _Dir[MAX_PATH];
            T _Driver[sizeof(T) * 2];
            T word_dir[MAX_PATH];
            ::GetModuleFileName(nullptr, current_exe_fname, MAX_PATH);
#if defined(_UNICODE) or defined(UNICODE)
            ::_wsplitpath_s(current_exe_fname, _Driver, sizeof(T) * 2, _Dir, MAX_PATH, nullptr, 0, nullptr, 0);
#else
            ::_splitpath_s(current_exe_fname, _Driver, sizeof(T) * 2, _Dir, MAX_PATH, nullptr, 0, nullptr, 0);
#endif
            ::wsprintf(word_dir, _T("%s%s"), _Driver, _Dir);
            return std::basic_string<T, std::char_traits<T>, std::allocator<T>>(word_dir);
        }

        /**
        *@brief dll下根据运行时环境获取子dll的绝对加载路径
        *@param g_dllModule DllMain函数的DLL_THREAD_ATTACH下通过g_dllModule = hModule获取
        *@param sub_dll_name 子dll名 xxx.dll
        *@return 子dll的绝对加载路径
        */
        template<typename T>
        auto getSubDllFileName(const HMODULE& g_dll_module, const T* sub_dll_name) noexcept
        {
            T current_dll_fname[MAX_PATH];
            T _Dir[MAX_PATH];
            T _Driver[sizeof(T) * 2];
            T sub_dll_fname[MAX_PATH];
            ::GetModuleFileName(g_dll_module, current_dll_fname, MAX_PATH);
#if defined(_UNICODE) or defined(UNICODE)
            ::_wsplitpath_s(current_dll_fname, _Driver, sizeof(T) * 2, _Dir, MAX_PATH, NULL, 0, NULL, 0);
#else
            ::_splitpath_s(current_dll_fname, _Driver, sizeof(T) * 2, _Dir, MAX_PATH, NULL, 0, NULL, 0);
#endif
            ::wsprintf(sub_dll_fname, _T("%s%s%s"), _Driver, _Dir, sub_dll_name);
            return std::basic_string<T, std::char_traits<T>, std::allocator<T>>(sub_dll_fname);
        };

        /**
        *@brief 运行时目录下搜索DLL及其依赖项
        */
        template<typename T>
        HMODULE loadSubDll(const HMODULE& g_dll_module, const T* sub_dll_name) noexcept
        {
            std::basic_string<T, std::char_traits<T>, std::allocator<T>> sub_dll_path = getSubDllFileName<T>(g_dll_module, sub_dll_name);
            return LoadLibraryEx(sub_dll_path.data(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }

        /**
        *@brief 指定目录搜索DLL及其依赖项
        */
        static HMODULE loadSubDll(const std::wstring& sub_dll_dir, const std::wstring& sub_dll_name) noexcept
        {
            ::AddDllDirectory(sub_dll_dir.data());
            return LoadLibraryExW(sub_dll_name.data(), nullptr, LOAD_LIBRARY_SEARCH_USER_DIRS);
        }

        /**
        *@brief 在应用程序的安装目录中搜索DLL及其依赖项
        */
        template<typename T>
        HMODULE loadSubDll(const T* sub_dll_name) noexcept
        {
            std::basic_string<T, std::char_traits<T>, std::allocator<T>> _sub_dll_name = sub_dll_name;
            return LoadLibraryEx(_sub_dll_name.data(), nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);
        }

        /**
        *@brief 外部调用dll函数获取类实例指针或函数地址
        */
        template<class interfaceCls>
        class ProcAddress
        {
        public:
            typedef std::shared_ptr<interfaceCls>(*func_type_name)();

            /**
            *@brief 外部调用dll函数获取函数地址
            */
            func_type_name getAddress(const HMODULE& dll_module, const std::string& func_name) noexcept
            {
                return (func_type_name)GetProcAddress(dll_module, func_name.data());
            }

            /**
            *@brief 外部调用dll函数获取类实例指针
            *@return 若失败返回nullptr
            */
            std::shared_ptr<interfaceCls> getPtr(const HMODULE& dll_module, const std::string& func_name) noexcept
            {
                func_type_name func = getAddress(dll_module, func_name.data());
                if (!func) {
                    return nullptr;
                }
                std::shared_ptr<interfaceCls> p = nullptr;
                try {
                    p = func();
                }
                catch (const std::exception&) {
                    return nullptr;
                }
                return p;
            }
        };

    } // namespace windows

} // namespace common

#endif // _COMMON_WINDOWS_HPP_
