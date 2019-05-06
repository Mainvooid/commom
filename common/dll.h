/*
@brief dll helper
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _DLL_H_
#define _DLL_H_

#include <common/precomm.h>
#include <memory>

#if defined(_WIN32) && !defined(DLLAPI)
#define DLLAPI __declspec(dllexport)
#else
#define DLLAPI __declspec(dllimport)
#endif

namespace common{
    namespace dll {

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
    }
}

#endif // _DLL_H_
