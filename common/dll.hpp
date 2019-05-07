/*
@brief dll helper
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#if !defined(_COMMON_DLL_HPP_) && defined(_WIN32)
#define _COMMON_DLL_HPP_

#include <common/precomm.hpp>
#include <memory>
#include <windows.h>

#ifndef DLLAPI
#define DLLAPI __declspec(dllexport)
#else
#define DLLAPI __declspec(dllimport)
#endif

namespace common {

    namespace dll {

        ///DLL导出

        /**
        *@brief 用于dll导出接口类的实现类指针,自动管理内存,以防范Cross-Dll问题
        */
        template<class interfaceCls, class implCls>
        std::shared_ptr<interfaceCls> getClsPtr() noexcept
        {
            std::shared_ptr<interfaceCls> ptr;
            ptr.reset(new implCls);
            return ptr;
        }

        ///DLL导入

        /**
        *@brief dll下根据运行时环境获取子dll的绝对加载路径
        *@param g_dllModule DllMain函数的DLL_THREAD_ATTACH下通过g_dllModule = hModule获取
        *@param sub_dll_name 子dll名 xxx.dll
        *@return 子dll的绝对加载路径
        */
        static std::wstring getSubDllFileName(const HMODULE& g_dllModule,const std::wstring& sub_dll_name) noexcept
        {
            wchar_t current_dll_fname[MAX_PATH];
            wchar_t _Dir[MAX_PATH];
            wchar_t _Driver[sizeof(wchar_t) * 2];
            wchar_t sub_dll_fname[MAX_PATH];
            ::GetModuleFileNameW(g_dllModule, current_dll_fname, MAX_PATH);
            ::_wsplitpath_s(current_dll_fname, _Driver, sizeof(wchar_t) * 2,
                _Dir, MAX_PATH, NULL, 0, NULL, 0);
            ::wsprintfW(sub_dll_fname, _T("%s%s%s"), _Driver, _Dir, sub_dll_name);
            return sub_dll_fname;
        };

        /**
        *@brief 运行时目录下搜索DLL及其依赖项
        */
        static HMODULE loadSubDll(const HMODULE& g_dllModule,const std::wstring& sub_dll_name) noexcept
        {
            std::wstring sub_dll_path = getSubDllFileName(g_dllModule, sub_dll_name);
            return LoadLibraryEx(sub_dll_path.data(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }

        /**
        *@brief 指定目录搜索DLL及其依赖项
        */
        static HMODULE loadSubDll(const std::wstring& sub_dll_dir,const std::wstring& sub_dll_name) noexcept
        {
            ::AddDllDirectory(sub_dll_dir.data());
            return LoadLibraryEx(sub_dll_name.data(), nullptr, LOAD_LIBRARY_SEARCH_USER_DIRS);
        }

        /**
        *@brief 在应用程序的安装目录中搜索DLL及其依赖项
        */
        static HMODULE loadSubDll(const std::wstring& sub_dll_name) noexcept
        {
            return LoadLibraryEx(sub_dll_name.data(), nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);
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
            func_type_name getAddress(const HMODULE& dll,const std::string& func_name) noexcept
            {
                return GetProcAddress(dll, func_name.data());
            }

            /**
            *@brief 外部调用dll函数获取类实例指针
            *@return 若失败返回nullptr
            */
            std::shared_ptr<interfaceCls> getPtr(const HMODULE& dll,const std::string& func_name) noexcept
            {
                func_type_name func = getAddress(dll, func_name.data());
                if (!func)
                {
                    return nullptr;
                }
                std::shared_ptr<interfaceCls> p = nullptr;
                try
                {
                    p = func();
                }
                catch (const std::exception&)
                {
                    return nullptr;
                }
                return p;
            }
        };

    } // namespace dll

} // namespace common

#endif // _COMMON_DLL_HPP_
