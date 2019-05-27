/*
@brief dll helper
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifdef HAVE_WINDOWS

#if !defined(_COMMON_WINDOWS_HPP_) && defined(_WIN32)
#define _COMMON_WINDOWS_HPP_

#include <common/precomm.hpp>
#include <memory>
#include <iostream>
#include <windows.h>
#include <d3d11.h>
#include <d3dx11.h>

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
            T work_dir[MAX_PATH];
            ::GetModuleFileName(nullptr, current_exe_fname, MAX_PATH);
#if defined(_UNICODE) or defined(UNICODE)
            ::_wsplitpath_s(current_exe_fname, _Driver, sizeof(T) * 2, _Dir, MAX_PATH, nullptr, 0, nullptr, 0);
#else
            ::_splitpath_s(current_exe_fname, _Driver, sizeof(T) * 2, _Dir, MAX_PATH, nullptr, 0, nullptr, 0);
#endif
            ::wsprintf(work_dir, _T("%s%s"), _Driver, _Dir);
            return std::basic_string<T, std::char_traits<T>, std::allocator<T>>(work_dir);
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

        ///directX

#ifdef HAVE_DIRECTX

        HRESULT createD3D11Device(ID3D11Device** ppDevice, IDXGIAdapter* pAdapter = nullptr, ID3D11DeviceContext** ppImmediateContext = nullptr) {
            UINT createDeviceFlags = 0;

#if defined(_DEBUG) or defined(DEBUG)
            createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
            createDeviceFlags |= D3D11_CREATE_DEVICE_BGRA_SUPPORT;

            D3D_FEATURE_LEVEL featureLevel;
            HRESULT  hr = D3D11CreateDevice(
                pAdapter,                  //IDXGIAdapter* 默认显示适配器
                D3D_DRIVER_TYPE_HARDWARE,  //D3D_DRIVER_TYPE 驱动类型
                0,                         //HMODULE 不使用软件驱动
                createDeviceFlags,
                0,                         //若为nullptr则为默认特性等级，否则需要提供特性等级数组
                0,                         //特性等级数组的元素数目
                D3D11_SDK_VERSION,         //SDK版本
                ppDevice,                  //输出D3D设备
                &featureLevel,             //输出当前应用D3D特性等级
                ppImmediateContext);       //输出D3D设备上下文

            if (FAILED(hr) || featureLevel != D3D_FEATURE_LEVEL_11_0) {
                return S_FALSE;
            }
            return S_OK;
        }
#endif // HAVE_DIRECTX

    } // namespace windows

} // namespace common

#endif // _COMMON_WINDOWS_HPP_
#endif // HAVE_WINDOWS
