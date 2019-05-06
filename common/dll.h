/*
@brief dll helper
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#if !defined(_DLL_H_) && defined(_WIN32)
#define _DLL_H_

#include <common/precomm.h>
#include <memory>
#include <windows.h>

#ifndef DLLAPI
#define DLLAPI __declspec(dllexport)
#else
#define DLLAPI __declspec(dllimport)
#endif

namespace common {

    namespace dll {

        /**
        *@brief dll下根据运行时环境获取子dll的绝对加载路径
        *@param g_dllModule DllMain函数中DLL_THREAD_ATTACH通过g_dllModule = hModule获取
        *@param sub_dll_name 子dll名 xxx.dll
        *@return 子dll的绝对加载路径
        */
        template<bool flag = false>
        std::wstring getSubDllFileName(const HMODULE& g_dllModule, const std::wstring sub_dll_name) {
            wchar_t current_dll_fname[MAX_PATH];
            wchar_t _Dir[MAX_PATH];
            wchar_t _Driver[2];
            wchar_t sub_dll_fname[MAX_PATH];
            ::GetModuleFileNameW(g_dllModule, current_dll_fname, MAX_PATH);
            ::_wsplitpath_s(current_dll_fname, _Driver, 2, _Dir, MAX_PATH, NULL, 0, NULL, 0);
            ::wsprintfW(sub_dll_fname, _T("%s%s%s"), _Driver, _Dir, std::move(sub_dll_name));
            return sub_dll_fname;
        };

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

    } // namespace dll

} // namespace common

#endif // _DLL_H_
