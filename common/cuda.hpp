/*
@brief cuda common, cuda_d3d11_interop
@author guobao.v@gmail.com
*/
#ifndef _COMMON_CUDA_HPP_
#define _COMMON_CUDA_HPP_
#include <common/debuglog.hpp>
#include <common/windows.hpp>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <driver_types.h>

#ifdef HAVE_CUDA_KERNEL
#include <common/cuda/texture_reference.hpp>
#endif

#if defined(_WIN32) && !defined(_WIN64)
#error cuda need win64 in windows , undefine macro(HAVE_CUDA) to block
#else
#pragma comment(lib,"cudart.lib")
#endif

#ifdef HAVE_DIRECTX
#include <cuda_d3d11_interop.h>
#include <wrl/client.h>
#include <dxgi.h>
#include <D3DX11async.h>
#include <D3DCompiler.h>

#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib,"dxerr.lib")

#ifdef HAVE_OPENCV
#include <common/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#endif // HAVE_OPENCV

#endif // HAVE_DIRECTX

/**
  @addtogroup common
  @{
    @defgroup cuda cuda - cuda utilities
  @}
*/
namespace common {
    /// @addtogroup common
    /// @{
    namespace cuda {
        /// @addtogroup cuda
        /// @{

        /**
        @defgroup kernel kernel - cuda host and device functions
        */
        namespace kernel {}

        /**
        @brief cudaError_t检查,若失败会中断程序
        */
        template<typename T = char>
        struct CheckCudaRet {
            void operator()(cudaError_t result, char const *const func, const char *const file, int const line) {
                if (result) {
                    common::LOGE(cudaGetErrorName(result), func, file, line);
                    cudaDeviceReset();// Make sure we call CUDA Device Reset before exiting
                    exit(EXIT_FAILURE);
                }
            }
        };

        /**
        @def checkCudaRet
        @brief cuda函数返回值检查,若失败会中断程序
        */
#ifndef checkCudaRet
#define checkCudaRet(val) common::cuda::CheckCudaRet<>()((val), #val, __FILE__, __LINE__)
#endif

        /**
        @brief CUDA设备检查
        */
        static cudaError_t checkCUDADevice()
        {
            int deviceCount = 0;
            checkCudaRet(cudaGetDeviceCount(&deviceCount));
            if (deviceCount == 0) {
                return cudaError::cudaErrorNoDevice;
            }
            return cudaError::cudaSuccess;
        }

        /**
        @brief 统计CUDA函数体的gpu时间(主要用于统计内核函数调用开销，包含CPU代码的时间可能是不完整的)
        @param Fn 函数对象,可用匿名函数包装代码片段来计时
        @param args 函数参数
        @return 单位ms
        */
        template<typename R, typename ...FArgs, typename ...Args>
        float getCudaFnDuration(std::function<R(FArgs...)> Fn, Args&... args) {
            float duration;
            cudaEvent_t start, stop;
            checkCudaRet(cudaEventCreate(&start));
            checkCudaRet(cudaEventCreate(&stop));
            checkCudaRet(cudaEventRecord(start, 0));
            Fn(args...);
            checkCudaRet(cudaEventRecord(stop, 0));
            checkCudaRet(cudaEventSynchronize(stop));
            checkCudaRet(cudaEventElapsedTime(&duration, start, stop));
            checkCudaRet(cudaEventDestroy(start));
            checkCudaRet(cudaEventDestroy(stop));
            return duration;
        }

#ifdef HAVE_DIRECTX
        // cuda_d3d11_interop

        /**
        @brief DX11和CUDA共享的纹理数据结构
        @todo 考虑基类使用ID3D11Resource
        */
        template<typename T, int texType, enum cudaTextureReadMode mode = cudaReadModeElementType>
        struct shared_texture_t
        {//TODO 考虑基类使用ID3D11Resource
#ifdef HAVE_CUDA_KERNEL
            /**@note 暂只支持uchar4和float4*/
            shared_texture_t() {
                cuda_get_texture_reference<T, texType, cudaReadModeElementType>()(&texture_ref);
                //cuda_get_texture_reference_2d_uchar4(&texture_ref);
            }
            ~shared_texture_t() {};
#endif
            const textureReference* texture_ref = nullptr;//纹理参考系引用,需要初始化
            cudaGraphicsResource* cuda_resource = nullptr;
            cudaArray* cuda_array = nullptr;
            cudaChannelFormatDesc cuda_array_desc = cudaCreateChannelDesc<T>();
            size_t pitch;
        };
        template<typename T, enum cudaTextureReadMode mode = cudaReadModeElementType>
        struct shared_texture_1d_t : shared_texture_t<T, cudaTextureType1D, mode>
        {
            ~shared_texture_1d_t() { p_d3d11_texture_1d.Reset(); };
            ComPtr<ID3D11Texture1D> p_d3d11_texture_1d;
            UINT width;
        };
        template<typename T, enum cudaTextureReadMode mode = cudaReadModeElementType>
        struct shared_texture_2d_t : shared_texture_t<T, cudaTextureType2D, mode>
        {
            ~shared_texture_2d_t() { p_d3d11_texture_2d.Reset(); };
            ComPtr<ID3D11Texture2D> p_d3d11_texture_2d;
            UINT width;
            UINT height;
        };
        template<typename T, enum cudaTextureReadMode mode = cudaReadModeElementType>
        struct shared_texture_3d_t : shared_texture_t<T, cudaTextureType3D, mode>
        {
            ~shared_texture_3d_t() { p_d3d11_texture_3d.Reset(); };
            ComPtr<ID3D11Texture3D> p_d3d11_texture_3d;
            UINT width;
            UINT height;
            UINT depth;
        };

        /**
        @brief CUDA获取D3D11设备适配器
        @exception std::runtime_error 无法成功获取设备适配器
        */
        static void getD3D11Adapter(IDXGIAdapter** pAdapter) noexcept(false)
        {
            ComPtr<IDXGIFactory> pFactory;
            HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(pFactory.GetAddressOf()));
            if (FAILED(hr)) {
                throw std::runtime_error("No DXGI Factory created");
            }

            ComPtr<IDXGIAdapter> _pAdapter;
            for (UINT i = 0; *pAdapter == nullptr; ++i)
            {
                //获取一个候选DXGI适配器
                hr = pFactory->EnumAdapters(i, _pAdapter.GetAddressOf());
                if (FAILED(hr)) { break; }

                //查询是否存在相应的计算设备
                int cuDevice;
                checkCudaRet(cudaD3D11GetDevice(&cuDevice, _pAdapter.Get()));
                Release_s(*pAdapter);
                *pAdapter = _pAdapter.Get();
                (*pAdapter)->AddRef();
            }
            if (*pAdapter == nullptr) {
                throw std::runtime_error("No IDXGIAdapter");
            }
        }

#ifdef HAVE_OPENCV
        /**
        @brief d3d11 texture2d 与opencv gpumat 互转
        @attention 目前只测试了uchar4( D3D R8G8B8A8 <-> opencv CV_8UC4 ),其他暂未支持
        @note texture2d_to_gpumat,gpumat_to_texture2d需配合其对应的绑定解绑方法调用,并且不能跨线程.
        */
        template<typename T = uchar4>
        class texture2d_cvt_gpumat
        {
        public:
            texture2d_cvt_gpumat() {
                getD3D11Adapter(mp_cuda_capable_adater.GetAddressOf());
                HRESULT hr = windows::createD3D11Device(mp_d3d11_device.GetAddressOf(), mp_cuda_capable_adater.Get());
                if (FAILED(hr)) LOGF_("createD3D11Device Failed!");
            };
            ~texture2d_cvt_gpumat() {
                mp_cuda_capable_adater.~ComPtr();
                mt_texture2d_to_gpumat.~shared_texture_2d_t();
                mt_gpumat_to_texture2d.~shared_texture_2d_t();
                mp_d3d11_device.~ComPtr();
            };

            /**
            @brief texture2d_to_gpumat 之前调用一次,注册映射绑定资源
            @param[in] p_src_texture 源D3D11纹理(R8G8B8A8),若地址/DESC发生改变需要解绑后重新绑定
            @see texture2d_to_gpumat
            */
            bool bind_for_texture2d_to_gpumat(ID3D11Texture2D* p_src_texture) {
                HRESULT hr = common::windows::texture2d_to_texture2d(p_src_texture, mt_texture2d_to_gpumat.p_d3d11_texture_2d.ReleaseAndGetAddressOf(), mp_d3d11_device.Get());
                if (FAILED(hr)) { return false; }
                //注册Direct3D 11资源以供CUDA访问
                checkCudaRet(cudaGraphicsD3D11RegisterResource(&mt_texture2d_to_gpumat.cuda_resource, mt_texture2d_to_gpumat.p_d3d11_texture_2d.Get(), cudaGraphicsRegisterFlagsNone));
                //映射图形资源以供CUDA访问,不能跨线程调用
                checkCudaRet(cudaGraphicsMapResources(1, &mt_texture2d_to_gpumat.cuda_resource));
                //获取一个数组，通过该数组访问映射图形资源的子资源
                checkCudaRet(cudaGraphicsSubResourceGetMappedArray(&mt_texture2d_to_gpumat.cuda_array, mt_texture2d_to_gpumat.cuda_resource, 0, 0));
                //绑定纹理参考系
                checkCudaRet(cudaBindTextureToArray(mt_texture2d_to_gpumat.texture_ref, mt_texture2d_to_gpumat.cuda_array, &mt_texture2d_to_gpumat.cuda_array_desc));
                return true;
            }

            /**
            @brief texture2d到gpumat的转换
            @pre 调用前通过bind_for_texture2d_to_gpumat方法绑定一次纹理资源,使用完成后通过unbind_for_texture2d_to_gpumat解绑
            @param[out] dst_gpumat  目标矩阵,格式取决于p_src_texture(R8G8B8A8)
            @see bind_for_texture2d_to_gpumat , unbind_for_texture2d_to_gpumat
            */
            bool texture2d_to_gpumat(cv::cuda::GpuMat& dst_gpumat) {
                D3D11_TEXTURE2D_DESC desc;
                mt_texture2d_to_gpumat.p_d3d11_texture_2d->GetDesc(&desc);
                dst_gpumat.create(desc.Height, desc.Width, CV_MAKETYPE(CV_8U, sizeof(T)));
                checkCudaRet(cudaMemcpy2DFromArray(dst_gpumat.data, dst_gpumat.step,
                    mt_texture2d_to_gpumat.cuda_array, 0, 0, dst_gpumat.cols * sizeof(T), dst_gpumat.rows, cudaMemcpyDeviceToDevice));
                return true;
            };
            /**@overload*/
            bool texture2d_to_gpumat(ID3D11Texture2D* p_src_texture, cv::cuda::GpuMat& dst_gpumat) {
                bool flag = bind_for_texture2d_to_gpumat(p_src_texture);
                flag = texture2d_to_gpumat(dst_gpumat);
                unbind_for_texture2d_to_gpumat();
                return flag;
            }

            /**
            @brief texture2d_to_gpumat 结束后调用一次,解除注册映射绑定资源
            @see texture2d_to_gpumat
            */
            void unbind_for_texture2d_to_gpumat() {
                checkCudaRet(cudaUnbindTexture(mt_texture2d_to_gpumat.texture_ref));
                checkCudaRet(cudaGraphicsUnmapResources(1, &mt_texture2d_to_gpumat.cuda_resource));
                checkCudaRet(cudaGraphicsUnregisterResource(mt_texture2d_to_gpumat.cuda_resource));
            }

            /**
            @brief 注册D3D11资源以供CUDA访问,此调用可能具有高开销，并且不应在交互式应用程序中每帧调用
            @see gpumat_to_texture2d
            */
            bool bind_for_gpumat_to_texture2d(int width, int height) {
                D3D11_TEXTURE2D_DESC desc;
                windows::createTextureDesc(desc, width, height,
                    DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED);
                HRESULT hr = mp_d3d11_device->CreateTexture2D(&desc, nullptr, mt_gpumat_to_texture2d.p_d3d11_texture_2d.ReleaseAndGetAddressOf());
                if (FAILED(hr)) { return false; }
                //注册Direct3D 11资源以供CUDA访问
                checkCudaRet(cudaGraphicsD3D11RegisterResource(&mt_gpumat_to_texture2d.cuda_resource, mt_gpumat_to_texture2d.p_d3d11_texture_2d.Get(), cudaGraphicsRegisterFlagsNone));
                //映射图形资源以供CUDA访问,不能跨线程调用
                checkCudaRet(cudaGraphicsMapResources(1, &mt_gpumat_to_texture2d.cuda_resource));
                //获取一个数组，通过该数组访问映射图形资源的子资源
                checkCudaRet(cudaGraphicsSubResourceGetMappedArray(&mt_gpumat_to_texture2d.cuda_array, mt_gpumat_to_texture2d.cuda_resource, 0, 0));
                checkCudaRet(cudaBindTextureToArray(mt_gpumat_to_texture2d.texture_ref, mt_gpumat_to_texture2d.cuda_array, &mt_gpumat_to_texture2d.cuda_array_desc));
                return true;
            }

            /**
            @brief gpumat到texture2d的转换
            @pre 此方法使用前需要注册一次资源,使用结束后需要取消注册,并且同一分辨率只能注册一次,若分辨率更改需要重新注册.
            @param[in] src_gpumat 源矩阵,应输入RGBA格式
            @param[out] pp_dst_texture 目标纹理对象, R8G8B8A8
            @param[in] p_dst_device 目标纹理对象的设备
            @note CUDA要求不能频繁注册/解注册资源.否则会导致显存out of memory.
            @see bind_for_gpumat_to_texture2d , unbind_for_gpumat_to_texture2d
            */
            bool gpumat_to_texture2d(cv::cuda::GpuMat src_gpumat, ID3D11Texture2D** pp_dst_texture, ID3D11Device* p_dst_device,cv::cuda::Stream& stream) {
                if (src_gpumat.channels() == 1) {
                    cv::cuda::cvtColor(src_gpumat, src_gpumat, cv::COLOR_GRAY2RGBA, 0, stream);
                }
                if (src_gpumat.channels() == 3) {
                    cv::cuda::cvtColor(src_gpumat, src_gpumat, cv::COLOR_BGR2RGBA, 0, stream);
                }
                stream.waitForCompletion();
                checkCudaRet(cudaMemcpy2DToArray(mt_gpumat_to_texture2d.cuda_array, 0, 0, src_gpumat.data, src_gpumat.step,
                    src_gpumat.cols * sizeof(T), src_gpumat.rows, cudaMemcpyDeviceToDevice));
                HRESULT hr = common::windows::texture2d_to_texture2d(mt_gpumat_to_texture2d.p_d3d11_texture_2d.Get(), pp_dst_texture, p_dst_device);
                if (FAILED(hr)) { return false; }
                return true;
            };
            /**
            @brief gpumat_to_texture2d 调用结束后解绑
            @see gpumat_to_texture2d
            */
            void unbind_for_gpumat_to_texture2d() {
                checkCudaRet(cudaUnbindTexture(mt_gpumat_to_texture2d.texture_ref));
                checkCudaRet(cudaGraphicsUnmapResources(1, &mt_gpumat_to_texture2d.cuda_resource));
                checkCudaRet(cudaGraphicsUnregisterResource(mt_gpumat_to_texture2d.cuda_resource));
            }
        private:
            ComPtr<ID3D11Device> mp_d3d11_device;              ///< D3D11设备
            ComPtr<IDXGIAdapter> mp_cuda_capable_adater;       ///< CUDA需要的DX适配器
            shared_texture_2d_t<T> mt_texture2d_to_gpumat;     ///< 共享纹理对象for texture2d_to_gpumat
            shared_texture_2d_t<T> mt_gpumat_to_texture2d;     ///< 共享纹理对象for gpumat_to_texture2d
        };

#endif // HAVE_OPENCV
#endif // HAVE_DIRECTX
        /// @}
    } // namespace cuda
    /// @}
} // namespace common

#endif // _COMMON_CUDA_HPP_