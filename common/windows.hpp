/*
@brief dll helper
@author guobao.v@gmail.com
*/
#if !defined(_COMMON_WINDOWS_HPP_) && defined(_WIN32)
#define _COMMON_WINDOWS_HPP_

#include <common/precomm.hpp>
#include <windows.h>

#ifdef HAVE_DIRECTX
#include <d3d9.h>
#include <d3dx9.h>
#include <d3d11.h>
#include <d3dx11.h>
#include <wrl\client.h>
#include <DirectXMath.h>
#include <D3DX11async.h>
#include <D3DCompiler.h>
using Microsoft::WRL::ComPtr;

#if defined(_DEBUG) || defined(DEBUG)
#include <d3dcommon.h>
#endif

#pragma comment ( lib, "dxguid.lib") 
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

/**
  @addtogroup common
  @{
    @defgroup windows windows - windows utilities
  @}
*/
namespace common {
    /// @addtogroup common
    /// @{
    namespace windows {
        /// @addtogroup windows
        /// @{

        template <typename T>
        inline auto to_tchar(T str) {
            return common::tvalue<TCHAR>(codecvt::to_ansi(str),
                codecvt::to_unicode(str));
        }

        template <typename T>
        inline auto to_tchar(const T* str) {
            std::basic_string<T, std::char_traits<T>, std::allocator<T>> _str = str;
            return to_tchar(_str);
        }

        /**
         @brief 包装CRITICAL_SECTION 在win上效率高一些
        */
        class win_mutex
        {
        private:
            CRITICAL_SECTION _lock;
        public:
            win_mutex() { InitializeCriticalSection(&_lock); };
            ~win_mutex() { DeleteCriticalSection(&_lock); };
            void lock() { EnterCriticalSection(&_lock); };
            bool trylock() { return TryEnterCriticalSection(&_lock); };
            void unlock() { LeaveCriticalSection(&_lock); };
        };

        //DLL导出

        /**
        @brief 用于dll导出接口类的实现类指针,自动管理内存,以防范Cross-Dll问题
        */
        template<class interfaceCls, class implCls>
        std::shared_ptr<interfaceCls> getClsPtr() noexcept
        {
            return std::make_shared<implCls>();
        }

        //DLL导入

       /**
        @brief 返回工作目录
        */
        template<typename T>
        static auto getWorkDir() noexcept
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
        @brief dll下根据运行时环境获取子dll的绝对加载路径
        @param g_dll_module DllMain函数的DLL_THREAD_ATTACH下通过g_dllModule = hModule获取
        @param sub_dll_name 子dll名 xxx.dll
        @return 子dll的绝对加载路径
        */
        template<typename T>
        static auto getSubDllFileName(const HMODULE& g_dll_module, const T* sub_dll_name) noexcept
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
        @brief 返回最后一条错误信息
        */
        template<typename T = wchar_t/*or char*/>
        static auto getLastErrorString()
        {
            LPVOID p_buf;
            tvalue<T>(getFunction(FormatMessageA), getFunction(FormatMessageW))(
                FORMAT_MESSAGE_ALLOCATE_BUFFER |
                FORMAT_MESSAGE_FROM_SYSTEM |
                FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL,
                GetLastError(),
                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                (typename std::conditional_t<std::is_same_v<T, char>, LPSTR, LPWSTR>)&p_buf,
                0, NULL);
            std::basic_string<T, std::char_traits<T>, std::allocator<T>> msg =
                (typename std::conditional_t<std::is_same_v<T, char>, LPSTR, LPWSTR>)p_buf;
            return msg;
        }

        /**
        @brief 运行时目录下搜索DLL及其依赖项
        */
        template<typename T>
        static HMODULE loadSubDll(const HMODULE& g_dll_module, const T* sub_dll_name) noexcept
        {
            std::basic_string<T, std::char_traits<T>, std::allocator<T>> sub_dll_path = getSubDllFileName<T>(g_dll_module, sub_dll_name);
            return LoadLibraryEx(sub_dll_path.data(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        }

        /**
        @brief 指定目录搜索DLL及其依赖项
        */
        static HMODULE loadSubDll(const std::wstring& sub_dll_dir, const std::wstring& sub_dll_name) noexcept
        {
            ::AddDllDirectory(sub_dll_dir.data());
            return LoadLibraryExW(sub_dll_name.data(), nullptr, LOAD_LIBRARY_SEARCH_USER_DIRS);
        }

        /**
        @brief 在应用程序的安装目录中搜索DLL及其依赖项
        */
        template<typename T>
        static HMODULE loadSubDll(const T* sub_dll_name) noexcept
        {
            std::basic_string<T, std::char_traits<T>, std::allocator<T>> _sub_dll_name = sub_dll_name;
            return LoadLibraryEx(_sub_dll_name.data(), nullptr, LOAD_LIBRARY_SEARCH_APPLICATION_DIR);
        }

        /**
        @brief 外部调用dll函数获取类实例指针或函数地址
        */
        template<class interfaceCls>
        class ProcAddress
        {
        public:
            typedef std::shared_ptr<interfaceCls>(*func_type_name)();

            /**
            @brief 外部调用dll函数获取函数地址
            */
            func_type_name getAddress(const HMODULE& dll_module, const std::string& func_name) noexcept
            {
                return (func_type_name)GetProcAddress(dll_module, func_name.data());
            }

            /**
            @brief 外部调用dll函数获取类实例指针
            @return 若失败返回nullptr
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

        /**
        @brief 函数fps控制(运行次/s)
        */
        class FnFpsControl {
        public:
            FnFpsControl(int fps) {
                m_hNullEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
                ::QueryPerformanceFrequency(&m_lpFrequency);
                m_llFreq = m_lpFrequency.QuadPart;

                m_llAvgFreqPerFrm = (m_llFreq * 1000) / fps;
                m_llAvgDeltaFreqPerFrm = (m_llFreq * 1000) % fps;

                m_llTotalDeltaFreq = 0;
            };

            /**
            @brief 固定fps执行函数
            @return 帧等待时间 为负数时说明资源占用达到上限(不再限制等待)
            */
            template<typename R, typename ...FArgs, typename ...Args>
            int run(std::function<R(FArgs...)> Fn, Args&... args) {
                ::QueryPerformanceCounter(&m_lpFrequency);
                LONGLONG llPerStart = m_lpFrequency.QuadPart;

                Fn(args...);

                ::QueryPerformanceCounter(&m_lpFrequency);
                LONGLONG llUseFreq = (m_lpFrequency.QuadPart - llPerStart) * 1000;
                LONGLONG llCurWaitFreq = (m_llTotalDeltaFreq + m_llAvgFreqPerFrm + m_llAvgDeltaFreqPerFrm - llUseFreq);

                ::QueryPerformanceFrequency(&m_lpFrequency);
                m_llFreq = m_lpFrequency.QuadPart;

                int iCurWaitMS = static_cast<int>(llCurWaitFreq / m_llFreq);
                m_llTotalDeltaFreq = llCurWaitFreq % m_llFreq;

                if (iCurWaitMS > 0) {
                    ::timeBeginPeriod(1);
                    ::WaitForSingleObject(m_hNullEvent, iCurWaitMS);
                    ::timeEndPeriod(1);
                }

                return iCurWaitMS;
            }

        private:
            HANDLE m_hNullEvent;
            LARGE_INTEGER m_lpFrequency;
            LONGLONG m_llAvgFreqPerFrm;
            LONGLONG m_llAvgDeltaFreqPerFrm;
            LONGLONG m_llFreq;
            LONGLONG m_llTotalDeltaFreq;
        };

        //一般性directX函数

        /**
        @brief 检查D3D对象是否释放(一般放在device->Release()之前)
        @param pDevice 设备对象 需要设备开启标志:D3D11_CREATE_DEVICE_DEBUG
        */
        static HRESULT reportLiveDeviceObjects(ID3D11Device* pDevice)
        {
            ComPtr<ID3D11Debug> d3dDebug;
            HRESULT hr = pDevice->QueryInterface(__uuidof(ID3D11Debug), reinterpret_cast<void**>(d3dDebug.GetAddressOf()));
            if (SUCCEEDED(hr)) {
                hr = d3dDebug->ReportLiveDeviceObjects(D3D11_RLDO_DETAIL);
            }
            return hr;
        }

        /**
        @brief 创建D3D11设备
        @param[out] ppDevice   目标设备
        @param[in]  pAdapter   适配器
        @param[in]  deiverType 设备类型
        @param[out] ppImmediateContext 上下文
        @param[in]  createDeviceFlags 设备标识
        @note Release模式下也可以通过定义D3D11_DEVICE_DEBUG开启设备调试模式
        */
        static HRESULT createD3D11Device(ID3D11Device** ppDevice, IDXGIAdapter* pAdapter = nullptr,
            D3D_DRIVER_TYPE deiverType = D3D_DRIVER_TYPE::D3D_DRIVER_TYPE_UNKNOWN,
            ID3D11DeviceContext** ppImmediateContext = nullptr, UINT createDeviceFlags = D3D11_CREATE_DEVICE_BGRA_SUPPORT)
        {
#if defined(DEBUG) || defined(D3D11_DEVICE_DEBUG)
            createDeviceFlags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
            D3D_FEATURE_LEVEL featureLevel;
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

        /**
        @brief D3D11保存Texture到文件
        @param[in] pDevice     Device对象
        @param[in] pSrcTexture Texture对象
        @param[in] path        以(.png)结尾的路径
        @param[in] format      图片保存格式
        */
        template<typename T>
        static HRESULT saveTextureToFile(ID3D11Device *pDevice, ID3D11Resource* pSrcTexture, const T* path,
            D3DX11_IMAGE_FILE_FORMAT format = D3DX11_IFF_PNG)
        {
            ComPtr<ID3D11DeviceContext> ctx;
            pDevice->GetImmediateContext(ctx.GetAddressOf());
            return tvalue<T>(getFunction(D3DX11SaveTextureToFileA), getFunction(D3DX11SaveTextureToFileW))(ctx.Get(), pSrcTexture, format, path);
        }

        /**
        @brief D3D9保存Texture到文件
        @param[in] pSrcTexture Texture对象
        @param[in] path        以(.png)结尾的路径
        */
        template<typename T>
        static HRESULT saveTextureToFile(IDirect3DTexture9* pSrcTexture, const T* path) {
            return tvalue<T>(getFunction(D3DXSaveTextureToFileA), getFunction(D3DXSaveTextureToFileW))(path, D3DXIFF_PNG, pSrcTexture, nullptr);
        }

        /**
        @brief 从文件读取Texture
        @param[in]  pDevice     Device对象
        @param[out] pTexture2D  用于保存的纹理对象
        @param[in]  path        要读取的图像文件路径
        @param[in]  format      数据读取格式
        @param[in]  bindFlags   数据使用类型
        @param[in]  miscFlags   资源标识
        */
        template<typename T>
        static HRESULT loadTextureFromFile(ID3D11Device *pDevice, ID3D11Texture2D **pTexture2D, const T* path,
            DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM,
            UINT bindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE,
            UINT miscFlags = D3D11_RESOURCE_MISC_SHARED)
        {
            D3DX11_IMAGE_LOAD_INFO loadInfo;
            ZeroMemory(&loadInfo, sizeof(D3DX11_IMAGE_LOAD_INFO));
            loadInfo.BindFlags = bindFlags;
            loadInfo.Format = format;
            loadInfo.MipLevels = D3DX11_DEFAULT; //产生最大的mipmaps层 
            loadInfo.MipFilter = D3DX11_FILTER_LINEAR;
            loadInfo.MiscFlags = miscFlags;
            ComPtr<ID3DX11ThreadPump> pump;
            Release_s(*pTexture2D);
            return tvalue<T>(getFunction(D3DX11CreateTextureFromFileA), getFunction(D3DX11CreateTextureFromFileW))
                (pDevice, path, &loadInfo, pump.Get(), (ID3D11Resource**)pTexture2D, nullptr);
        }

        /**
        @brief 创建2D纹理配置
        @param[in] desc        Texture配置
        @param[in] width       宽度
        @param[in] height      高度
        @param[in] format      DXGI支持的数据格式
        @param[in] bindFlags   数据使用类型
        @param[in] miscFlags   资源标识
        */
        static void createTextureDesc(D3D11_TEXTURE2D_DESC &desc, UINT width, UINT height,
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

        /**
        @brief 不同device的共享纹理间的转换
        @param[in]  pSrcTexture2D  源纹理
        @param[out] ppDstTexture2D 目标纹理
        @param[in]  pDstDevice     目标设备
        @param[out] dst_pHandle    目标共享句柄
        */
        static HRESULT texture2d_to_texture2d(ID3D11Texture2D* pSrcTexture2D, ID3D11Texture2D** ppDstTexture2D, ID3D11Device* pDstDevice, HANDLE* dst_pHandle = nullptr)
        {
            D3D11_TEXTURE2D_DESC desc;
            pSrcTexture2D->GetDesc(&desc);

            ComPtr<ID3D11Device> p_src_device;
            pSrcTexture2D->GetDevice(p_src_device.GetAddressOf());
            HRESULT hr = S_OK;
            ComPtr <ID3D11Texture2D> p_new_src_texture;
            if ((desc.MiscFlags & D3D11_RESOURCE_MISC_SHARED) != D3D11_RESOURCE_MISC_SHARED) {
                ComPtr <ID3D11DeviceContext> p_ctx;
                p_src_device->GetImmediateContext(p_ctx.GetAddressOf());
                desc.MiscFlags |= D3D11_RESOURCE_MISC_SHARED;
                hr = p_src_device->CreateTexture2D(&desc, nullptr, p_new_src_texture.GetAddressOf());
                if (FAILED(hr)) { return hr; }
                p_ctx->CopyResource(p_new_src_texture.Get(), pSrcTexture2D);
                p_ctx->Flush();
            }
            else {
                p_new_src_texture = pSrcTexture2D;
            }
            ComPtr<IDXGIResource> pSharedResource;
            hr = p_new_src_texture.As(&pSharedResource);//QueryInterface
            if (FAILED(hr)) { return hr; }

            HANDLE handle;
            hr = pSharedResource->GetSharedHandle(&handle);
            if (FAILED(hr)) { return hr; }

            hr = pDstDevice->OpenSharedResource(handle, __uuidof(IDXGIResource), (void**)(pSharedResource.ReleaseAndGetAddressOf()));
            if (FAILED(hr)) { return hr; }

            Release_s(*ppDstTexture2D);
            hr = pSharedResource->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(ppDstTexture2D));
            if (FAILED(hr)) { return hr; }

            if (dst_pHandle != nullptr) { *dst_pHandle = handle; }
            return S_OK;
        }

        // dxgi_format_convert

        /**
        @brief 顶点位置
        */
        struct VertexPos
        {
            DirectX::XMFLOAT3 pos;
            DirectX::XMFLOAT2 Tex;
        };

        /**
        @brief 像素着色器对象
        */
        struct DXGI_D3D11_PIXEL_SHADER_OBJ
        {
            Microsoft::WRL::ComPtr<ID3D11VertexShader> solidColorVS;
            Microsoft::WRL::ComPtr<ID3D11PixelShader> solidColorPS;
            Microsoft::WRL::ComPtr<ID3D11InputLayout> inputLayout;
            Microsoft::WRL::ComPtr<ID3D11Buffer> vertexBuffer;
            Microsoft::WRL::ComPtr<ID3D11SamplerState> colorMapSampler;
            void Reset() {
                solidColorVS.Reset();
                solidColorPS.Reset();
                inputLayout.Reset();
                vertexBuffer.Reset();
                colorMapSampler.Reset();
            }
        };

        /**
        @brief D3D11格式转换类
        */
        class DXGI_D3D11_FORMAT_BGRA2RGBA
        {
        public:
            DXGI_D3D11_FORMAT_BGRA2RGBA() {};

            /**
            *@brief 构造
            *@param D3D11Device D3D11设备
            *@param Width 图像宽度
            *@param Height 图像高度
            */
            void Create(ID3D11Device* D3D11Device, int Width, int Height) noexcept(false)
            {
                if (D3D11Device == nullptr) {
                    throw std::invalid_argument("ID3D11Device == nullptr");
                }
                mp_D3D11Device = D3D11Device;
                //获取上下文对象
                mp_D3D11Device->GetImmediateContext(mp_D3D11Context.ReleaseAndGetAddressOf());

                //创建交换纹理
                //TODO 这边固定配置为DXGI_FORMAT_R8G8B8A8_UNORM，理论上可以进行多种格式转换，尝试泛化此类
                D3D11_TEXTURE2D_DESC etDesc;
                ZeroMemory(&etDesc, sizeof(etDesc));
                etDesc.Width = Width;
                etDesc.Height = Height;
                etDesc.MipLevels = 1;											//纹理中最大的mipmap等级数(1,只包含最大的位图本身)
                etDesc.ArraySize = 1;											//纹理数目(可创建数组)
                etDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;						//DXGI支持的数据格式
                etDesc.SampleDesc.Count = 1;									//MSAA采样数(纹理通常不开启MSAA)
                etDesc.Usage = D3D11_USAGE_DEFAULT;								//指定数据的CPU/GPU访问权限(GPU读写)
                etDesc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;//数据使用类型
                etDesc.CPUAccessFlags = 0;										//CPU访问权限(0不需要)
                etDesc.MiscFlags = D3D11_RESOURCE_MISC_SHARED;					//资源标识

                HRESULT hr = mp_D3D11Device->CreateTexture2D(&etDesc, NULL, mp_ExchangeTexture.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    throw std::runtime_error("Create Exchange Texture Failed!");
                }

                //创建渲染视图
                D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
                renderTargetViewDesc.Format = etDesc.Format;
                renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;//渲染目标的资源类型
                renderTargetViewDesc.Texture2D.MipSlice = 0;

                hr = mp_D3D11Device->CreateRenderTargetView(mp_ExchangeTexture.Get(), &renderTargetViewDesc, mp_RTView.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    throw std::runtime_error("CreateRenderTargetView Failed!");
                }

                //创建着色器
                Create_Shader_Object();

                //创建缓冲区
                Prepare_Vector_Buffer();
            };

        private:

            /**
            *@brief 从文件编译D3D11着色器
            *@param pFunctionName 着色器入口点函数的名称
            *@param pProfile 着色器模型
            *@param ppShader 指向内存的指针，其中包含已编译的着色器，以及任何嵌入式调试和符号表信息
            */
            void Compile_D3D_Shader(const char* pFunctionName, const char* pProfile, ID3DBlob** ppShader)
            {
                DWORD shaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;

#ifdef _DEBUG
                shaderFlags |= D3DCOMPILE_DEBUG;
#endif

                ComPtr<ID3DBlob> pErrorMsgs;

                if (ppShader == nullptr) {
                    throw std::invalid_argument("ID3DBlob == nullptr");
                }

                static constexpr char colorMap_hlsl[] =
                    "Texture2D colorMap_ : register( t0 );\n"\
                    "SamplerState colorSampler_ : register(s0);\n"\
                    "struct VS_Input\n"\
                    "{\n"\
                    "    float4 pos : POSITION;\n"\
                    "    float2 tex0 : TEXCOORD0; \n"\
                    "};\n"\
                    "struct PS_Input\n"\
                    "{\n"\
                    "    float4 pos : SV_POSITION;\n"\
                    "    float2 tex0 : TEXCOORD0;\n"\
                    "};\n"\
                    "PS_Input VS_Main(VS_Input vertex)\n"\
                    "{\n"\
                    "    PS_Input vsOut = (PS_Input)0;\n"\
                    "    vsOut.pos = vertex.pos;\n"\
                    "    vsOut.tex0 = vertex.tex0;\n"\
                    "    return vsOut;\n"\
                    "}\n"\
                    "float4 PS_Main(PS_Input frag) : SV_TARGET\n"\
                    "{\n"\
                    "    return colorMap_.Sample(colorSampler_, frag.tex0);\n"\
                    "}\n";

                HRESULT hr = D3DX11CompileFromMemory(
                    colorMap_hlsl,
                    strlen(colorMap_hlsl),
                    "Memory", NULL, NULL,
                    pFunctionName,
                    pProfile,
                    shaderFlags, 0,
                    nullptr,
                    ppShader,
                    pErrorMsgs.GetAddressOf(),
                    nullptr);
                if (FAILED(hr)) {
                    throw std::invalid_argument((const char*)pErrorMsgs->GetBufferPointer());
                }
            };

            /**
            *@brief 创建D3D11顶点着色器,像素着色器
            */
            void Create_Shader_Object()
            {
                //顶点着色器对象
                ComPtr<ID3DBlob> vsBuffer;
                Compile_D3D_Shader("VS_Main", "vs_5_0", vsBuffer.GetAddressOf());

                //从已编译的着色器创建顶点着色器对象
                HRESULT hr = mp_D3D11Device->CreateVertexShader(
                    vsBuffer->GetBufferPointer(),		//已编译着色器的指针
                    vsBuffer->GetBufferSize(),			//已编译顶点着色器的大小
                    0,
                    m_PixelShaderObject.solidColorVS.ReleaseAndGetAddressOf());	//ID3D11VertexShader接口的指针的地址
                if (FAILED(hr)) {
                    throw std::invalid_argument("CreateVertexShader Failed!");
                }

                //输入布局对象

                D3D11_INPUT_ELEMENT_DESC solidColorLayout[] =
                {
                    { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },//输入数据是每顶点数据
                    { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D11_INPUT_PER_VERTEX_DATA, 0 }
                };

                unsigned int totalLayoutElements = ARRAYSIZE(solidColorLayout);
                //创建一个输入布局对象来描述输入:汇编程序阶段的输入缓冲区数据
                hr = mp_D3D11Device->CreateInputLayout(
                    solidColorLayout,
                    totalLayoutElements,
                    vsBuffer->GetBufferPointer(),
                    vsBuffer->GetBufferSize(),
                    m_PixelShaderObject.inputLayout.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    throw std::runtime_error("CreateInputLayout Failed!");
                }

                //创建像素着色器

                ComPtr<ID3DBlob> psBuffer;
                Compile_D3D_Shader("PS_Main", "ps_5_0", psBuffer.GetAddressOf());

                hr = mp_D3D11Device->CreatePixelShader(
                    psBuffer->GetBufferPointer(),
                    psBuffer->GetBufferSize(),
                    0,
                    m_PixelShaderObject.solidColorPS.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    throw std::runtime_error("CreatePixelShader Failed!");
                }

                //创建采样器状态对象

                D3D11_SAMPLER_DESC colorMapDesc;
                ZeroMemory(&colorMapDesc, sizeof(colorMapDesc));
                colorMapDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;//default 线性插值
                colorMapDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;//default:D3D11_TEXTURE_ADDRESS_CLAMP
                colorMapDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
                colorMapDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
                colorMapDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;//default 不比较采样数据
                colorMapDesc.MaxLOD = D3D11_FLOAT32_MAX;//default mipmap范围没有上限

                //创建一个采样器状态对象，该对象封装纹理的采样信息
                hr = mp_D3D11Device->CreateSamplerState(&colorMapDesc,
                    m_PixelShaderObject.colorMapSampler.ReleaseAndGetAddressOf());
                if (FAILED(hr)) {
                    throw std::runtime_error("CreateSamplerState Failed!");
                }
            };

            /**
            *@brief 创建缓冲区
            */
            void Prepare_Vector_Buffer()
            {
                if (m_PixelShaderObject.vertexBuffer) { return; }

                //创建缓冲区;
                //创建缩放矩形顶点缓存;
                //使用左手坐标系
                float fLeft = -1.0;
                float fRight = 1.0;
                float ftop = 1.0;
                float fBottom = -1.0;

                //计算缩放矩阵 原点(-1,1)宽度2->原点(-1,1)宽度1
                VertexPos vertices[] =
                {
                    { DirectX::XMFLOAT3(fRight, ftop, 1.0f), DirectX::XMFLOAT2(0.0f, 0.0f) },
                    { DirectX::XMFLOAT3(fRight,fBottom, 1.0f), DirectX::XMFLOAT2(0.0f, 1.0f) },
                    { DirectX::XMFLOAT3(fLeft, fBottom, 1.0f), DirectX::XMFLOAT2(-1.0f, 1.0f) },

                    { DirectX::XMFLOAT3(fLeft, fBottom, 1.0f), DirectX::XMFLOAT2(-1.0f, 1.0f) },
                    { DirectX::XMFLOAT3(fLeft,  ftop, 1.0f), DirectX::XMFLOAT2(-1.0f, 0.0f) },
                    { DirectX::XMFLOAT3(fRight, ftop, 1.0f), DirectX::XMFLOAT2(0.0f, 0.0f) },
                };

                D3D11_BUFFER_DESC vertexDesc;//表示缓冲区并提供创建缓冲区的方法
                ZeroMemory(&vertexDesc, sizeof(vertexDesc));
                vertexDesc.Usage = D3D11_USAGE_DEFAULT;
                vertexDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
                vertexDesc.ByteWidth = sizeof(VertexPos) * 6;

                D3D11_SUBRESOURCE_DATA subResourceData;//指定用于初始化子资源的数据
                ZeroMemory(&subResourceData, sizeof(subResourceData));
                subResourceData.pSysMem = vertices;//指向初始化数据的指针

                //创建缓冲区（顶点缓冲区，索引缓冲区或着色器常量缓冲区）
                HRESULT hr = mp_D3D11Device->CreateBuffer(
                    &vertexDesc,
                    &subResourceData,
                    m_PixelShaderObject.vertexBuffer.GetAddressOf());
                if (FAILED(hr)) {
                    throw std::runtime_error("CreateBuffer Failed!");
                }
            };

            /**
            *@brief 渲染到纹理
            *@param pSrc 源图像
            */
            void Render_To_Texture(ID3D11Texture2D* pSrc)
            {
                D3D11_VIEWPORT _OldVP;
                UINT nOldView = 1;
                mp_D3D11Context->RSGetViewports(&nOldView, &_OldVP);

                D3D11_TEXTURE2D_DESC src_desc;
                pSrc->GetDesc(&src_desc);

                D3D11_VIEWPORT vp;
                vp.Height = static_cast<FLOAT>(src_desc.Height);
                vp.Width = static_cast<FLOAT>(src_desc.Width);
                vp.MinDepth = 0.0f;
                vp.MaxDepth = 1.0f;
                vp.TopLeftX = 0;
                vp.TopLeftY = 0;
                mp_D3D11Context->RSSetViewports(1, &vp);
                mp_D3D11Context->OMSetRenderTargets(1, mp_RTView.GetAddressOf(), NULL);

                //填充渲染目标
                float color[4] = { 0,1.0,0,1.0 };
                mp_D3D11Context->ClearRenderTargetView(mp_RTView.Get(), color);

                ComPtr<ID3D11ShaderResourceView>  pSRView;//着色器资源视图
                HRESULT  hr = mp_D3D11Device->CreateShaderResourceView(pSrc, NULL, pSRView.GetAddressOf());
                if (FAILED(hr)) {
                    throw std::runtime_error("CreateShaderResourceView Failed!");
                }

                unsigned int stride = sizeof(VertexPos);
                unsigned int offset = 0;
                mp_D3D11Context->IASetInputLayout(m_PixelShaderObject.inputLayout.Get());
                mp_D3D11Context->IASetVertexBuffers(0, 1, m_PixelShaderObject.vertexBuffer.GetAddressOf(), &stride, &offset);
                mp_D3D11Context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                mp_D3D11Context->VSSetShader(m_PixelShaderObject.solidColorVS.Get(), 0, 0);
                mp_D3D11Context->PSSetShader(m_PixelShaderObject.solidColorPS.Get(), 0, 0);
                mp_D3D11Context->PSSetShaderResources(0, 1, pSRView.GetAddressOf());
                mp_D3D11Context->PSSetSamplers(0, 1, m_PixelShaderObject.colorMapSampler.GetAddressOf());
                mp_D3D11Context->Draw(6, 0);

                //RTT结束：恢复状态
                mp_D3D11Context->RSSetViewports(1, &_OldVP);
            };

        public:
            /**
            *@brief 进行DXGI格式转换(DXGI_FORMAT_B8G8R8A8_UNORM->DXGI_FORMAT_R8G8B8A8_UNORM)
            *@param pSrc 源图像
            *@param pDst 结果图像,传空指针接收
            */
            void Convert(ID3D11Texture2D* pSrc, ID3D11Texture2D*& pDst)
            {
                if (pSrc == nullptr) {
                    throw std::invalid_argument("ID3D11Texture2D == nullptr!");
                }
                Render_To_Texture(pSrc);
                pDst = mp_ExchangeTexture.Get();
            };

        private:
            Microsoft::WRL::ComPtr<ID3D11Device> mp_D3D11Device;        ///< D3D11设备
            Microsoft::WRL::ComPtr<ID3D11DeviceContext> mp_D3D11Context;///< D3D11设备上下文
            Microsoft::WRL::ComPtr<ID3D11Texture2D> mp_ExchangeTexture; ///< 转格式后的纹理
            DXGI_D3D11_PIXEL_SHADER_OBJ m_PixelShaderObject;            ///< 着色器对象
            Microsoft::WRL::ComPtr<ID3D11RenderTargetView> mp_RTView;   ///< 渲染目标视图
        };

#endif // HAVE_DIRECTX
        /// @}
    } // namespace windows
    /// @}
} // namespace common

#endif // _COMMON_WINDOWS_HPP_