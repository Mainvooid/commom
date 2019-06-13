/*
@brief dll helper
@author guobao.v@gmail.com
*/
#if !defined(_COMMON_WINDOWS_HPP_) && defined(_WIN32)
#define _COMMON_WINDOWS_HPP_

#include <common/precomm.hpp>
#include <memory>
#include <iostream>
#include <windows.h>

#ifdef HAVE_DIRECTX
#include <d3d9.h>
#include <d3dx9.h>
#include <d3d11.h>
#include <d3dx11.h>
#include <wrl\client.h>

#pragma comment ( lib, "d3d9.lib")
#pragma comment ( lib, "d3dx9.lib")
#pragma comment ( lib, "d3d11.lib")
#pragma comment ( lib, "d3dx11.lib")

#endif

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

#ifdef HAVE_DIRECTX

        ///一般性directX函数

        /*
        *@brief 创建D3D11设备
        *@param ppDevice 设备
        *@param pAdapter 适配器
        *@param ppImmediateContext 上下文
        */
        HRESULT createD3D11Device(ID3D11Device** ppDevice, IDXGIAdapter* pAdapter = nullptr,
            ID3D11DeviceContext** ppImmediateContext = nullptr)
        {
            UINT createDeviceFlags = 0;
#if defined(_DEBUG) or defined(DEBUG)
            createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
            createDeviceFlags |= D3D11_CREATE_DEVICE_BGRA_SUPPORT;

            D3D_FEATURE_LEVEL featureLevel;
            D3D_DRIVER_TYPE deiverType = D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_UNKNOWN;
            if (pAdapter == nullptr) {
                deiverType = D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_HARDWARE;
            }
            Release_s(*ppDevice);
            HRESULT  hr = D3D11CreateDevice(
                pAdapter,                  //IDXGIAdapter 默认显示适配器
                deiverType,                //D3D_DRIVER_TYPE 驱动类型
                0,                         //HMODULE 不使用软件驱动
                createDeviceFlags,
                nullptr,                   //若为nullptr则为默认特性等级，否则需要提供特性等级数组
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

        /*
        *@brief D3D11保存Texture到文件
        *@param pDevice    Device对象
        *@param pSrcTexture Texture对象
        *@param path       以(.png)结尾的路径
        */
        template<typename T>
        HRESULT saveTextureToFile(ID3D11Device *pDevice, ID3D11Resource* pSrcTexture, const T* path,
            D3DX11_IMAGE_FILE_FORMAT format = D3DX11_IFF_PNG)
        {
            Microsoft::WRL::ComPtr<ID3D11DeviceContext> ctx;
            pDevice->GetImmediateContext(ctx.GetAddressOf());
            return tvalue<T>(getFunction(D3DX11SaveTextureToFileA), getFunction(D3DX11SaveTextureToFileW))(ctx.Get(), pSrcTexture, format, path);
        }

        /*
        *@brief D3D9保存Texture到文件
        *@param pSrcTexture Texture对象
        *@param path       以(.png)结尾的路径
        */
        template<typename T>
        HRESULT saveTextureToFile(IDirect3DTexture9* pSrcTexture, const T* path) {
            return tvalue<T>(getFunction(D3DXSaveTextureToFileA), getFunction(D3DXSaveTextureToFileW))(path, D3DXIFF_PNG, pSrcTexture, nullptr);
        }

        /*
        *@brief 从文件读取Texture
        *@param d3dDevice  Device对象
        *@param path       要读取的图像文件路径
        *@param pTexture2D Texture2D对象
        *@param bindFlags  数据使用类型
        *@param format     读取格式
        */
        template<typename T>
        HRESULT loadTextureFromFile(ID3D11Device *pDevice, ID3D11Texture2D **pTexture2D, const T* path,
            DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
            UINT bindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE)
        {
            D3DX11_IMAGE_LOAD_INFO loadInfo;
            ZeroMemory(&loadInfo, sizeof(D3DX11_IMAGE_LOAD_INFO));
            loadInfo.BindFlags = bindFlags;
            loadInfo.Format = format;
            loadInfo.MipLevels = D3DX11_DEFAULT; //产生最大的mipmaps层
            loadInfo.MipFilter = D3DX11_FILTER_LINEAR;

            Microsoft::WRL::ComPtr<ID3DX11ThreadPump> pump;
            Release_s(*pTexture2D);
            return tvalue<T>(getFunction(D3DX11CreateTextureFromFileA), getFunction(D3DX11CreateTextureFromFileW))
                (pDevice, path, &loadInfo, pump.Get(), (ID3D11Resource**)pTexture2D, nullptr);
        }

        /*
        *@brief 创建2D纹理配置
        *@param textureDesc Texture配置
        *@param width       宽度
        *@param height      高度
        *@param format      DXGI支持的数据格式
        *@param bindFlags   数据使用类型
        */
        void createTextureDesc(D3D11_TEXTURE2D_DESC &desc, UINT width, UINT height,
            DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
            UINT bindFlags = D3D11_BIND_SHADER_RESOURCE,
            UINT miscFlags = 0)
        {
            ZeroMemory(&desc, sizeof(D3D11_TEXTURE2D_DESC));
            desc.Width = width;
            desc.Height = height;
            desc.MipLevels = 1;                          //纹理中最大的mipmap等级数(1,只包含最大的位图本身)
            desc.ArraySize = 1;                          //纹理数目(可创建数组)
            desc.Format = format;                        //DXGI支持的数据格式
            desc.SampleDesc.Count = 1;                   //MSAA采样数(纹理通常不开启MSAA)
            desc.Usage = D3D11_USAGE_DEFAULT;            //指定数据的CPU/GPU访问权限(GPU读写)
            desc.BindFlags = bindFlags;                  //数据使用类型
            desc.CPUAccessFlags = 0;                     //CPU访问权限(0不需要)
            desc.MiscFlags = miscFlags;                  //资源标识[D3D11_RESOURCE_MISC_SHARED]
        }

        /*
        *@brief 不同device的共享纹理间的转换
        *@param pSrcTexture2D    源纹理
        *@param pDstTexture2D    目标纹理
        *@param pDstDevice       目标设备
        *@param dst_handle       目标共享句柄
        */
        HRESULT texture2d_to_texture2d(ID3D11Texture2D* pSrcTexture2D, ID3D11Texture2D** ppDstTexture2D, ID3D11Device* pDstDevice, HANDLE* dst_handle = (HANDLE*)malloc(0))
        {
            D3D11_TEXTURE2D_DESC desc;
            pSrcTexture2D->GetDesc(&desc);

            Microsoft::WRL::ComPtr<ID3D11Device> p_src_device;
            pSrcTexture2D->GetDevice(p_src_device.GetAddressOf());
            HRESULT hr = S_OK;
            Microsoft::WRL::ComPtr <ID3D11Texture2D> p_new_src_texture;
            if ((desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED) != D3D11_RESOURCE_MISC_SHARED) {
                Microsoft::WRL::ComPtr <ID3D11DeviceContext> p_ctx;
                p_src_device->GetImmediateContext(&p_ctx);
                desc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
                hr = p_src_device->CreateTexture2D(&desc, nullptr, p_new_src_texture.GetAddressOf());
                if (FAILED(hr)) { return hr; }
                p_ctx->CopyResource(p_new_src_texture.Get(), pSrcTexture2D);
                p_ctx->Flush();
            }
            else {
                p_new_src_texture = pSrcTexture2D;
            }

            Microsoft::WRL::ComPtr<IDXGIResource> pSharedResource;
            hr = p_new_src_texture->QueryInterface(__uuidof(IDXGIResource), reinterpret_cast<void**>(pSharedResource.ReleaseAndGetAddressOf()));
            if (FAILED(hr)) { return hr; }

            hr = pSharedResource->GetSharedHandle(dst_handle);
            if (FAILED(hr)) { return hr; }
            hr = pDstDevice->OpenSharedResource(*dst_handle, __uuidof(IDXGIResource), (void**)(pSharedResource.ReleaseAndGetAddressOf()));
            if (FAILED(hr)) { return hr; }
            hr = pSharedResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)(ppDstTexture2D));
            if (FAILED(hr)) { return hr; }
            return S_OK;
        }

        /*
        *@brief 检查D3D对象是否释放(一般放在device->Release()之前)
        *@param pDevice 设备对象
        */
        HRESULT reportLiveDeviceObjects(ID3D11Device* pDevice)
        {
            Microsoft::WRL::ComPtr<ID3D11Debug> d3dDebug;
            HRESULT hr = pDevice->QueryInterface(__uuidof(ID3D11Debug), reinterpret_cast<void**>(d3dDebug.GetAddressOf()));
            if (SUCCEEDED(hr)) {
                hr = d3dDebug->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
            }
            return hr;
        }


#endif // HAVE_DIRECTX

    } // namespace windows

} // namespace common

#endif // _COMMON_WINDOWS_HPP_