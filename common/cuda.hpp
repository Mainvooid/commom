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
#include <cuda_d3d11_interop.h>

#ifdef HAVE_CUDA_KERNEL
#include <common/cuda/texture_reference.cuh>
#endif

#ifndef _WIN64
#error cuda need win64 , undefine macro to block : HAVE_CUDA
#else
#pragma comment(lib,"cudart.lib")
#endif

#ifdef HAVE_DIRECTX
#include <wrl/client.h>
#include <d3d11.h>
#include <d3dx11.h>
#include <dxgi.h>
#include <D3DX11async.h>
#include <D3DCompiler.h>

#pragma comment(lib,"d3d11.lib")
#pragma comment(lib,"d3dx11.lib")
#pragma comment(lib,"dxgi.lib")
#pragma comment(lib,"d3dcompiler.lib")
#pragma comment(lib,"dxerr.lib")

#ifdef HAVE_OPENCV
#include <opencv2/cudaimgproc.hpp> 
#endif // HAVE_OPENCV
#endif // HAVE_DIRECTX

namespace common {
    namespace cuda {

        /**
        *@brief cudaError_t检查,若失败会中断程序
        */
        template <typename T>
        void checkCudaRet(T result, char const *const func, const char *const file, int const line) {
            if (result) {
#if defined(_UNICODE) or defined(UNICODE)
                std::wostringstream oss;
#else
                std::ostringstream oss;
#endif
                oss << cudaGetErrorName(result);
                common::LOGE(oss.str(), func, file, line);
                cudaDeviceReset();
                // Make sure we call CUDA Device Reset before exiting
                exit(EXIT_FAILURE);
            }
        }
#define checkCudaRet(val) checkCudaRet((val), #val, __FILE__, __LINE__)

        /**
        *@brief CUDA设备检查
        */
        cudaError_t checkCUDADevice()
        {
            int deviceCount = 0;
            checkCudaRet(cudaGetDeviceCount(&deviceCount));
            if (deviceCount == 0) {
                return cudaError::cudaErrorNoDevice;
            }
            return cudaError::cudaSuccess;
        }

#ifdef HAVE_DIRECTX
        /// cuda_d3d11_interop

        /**
        *@brief DX11和CUDA共享的纹理数据结构
        */
        template<class T>
        struct shared_texture_t
        {
#ifdef HAVE_CUDA_KERNEL
            //TODO 暂只支持uchar4
            shared_texture_t() {
                checkCudaRet(cuda_get_texture_reference<uchar4>(&texture_ref));
            };
#endif
            //TODO 考虑基类使用ID3D11Resource
            const textureReference* texture_ref = nullptr;//纹理参考系引用,需要初始化
            cudaGraphicsResource* cuda_resource = nullptr;
            cudaArray* cuda_array = nullptr;
            cudaChannelFormatDesc cuda_array_desc = cudaCreateChannelDesc<T>();
            size_t pitch;
        };
        template<class T>
        struct shared_texture_1d_t : shared_texture_t<T>
        {
            ~shared_texture_1d_t() { p_d3d11_texture_1d.Reset(); };
            Microsoft::WRL::ComPtr<ID3D11Texture1D> p_d3d11_texture_1d;
            UINT width;
        };
        template<class T>
        struct shared_texture_2d_t : shared_texture_t<T>
        {
            Microsoft::WRL::ComPtr<ID3D11Texture2D> p_d3d11_texture_2d;
            UINT width;
            UINT height;
        };
        template<class T>
        struct shared_texture_3d_t : shared_texture_t<T>
        {
            Microsoft::WRL::ComPtr<ID3D11Texture3D> p_d3d11_texture_3d;
            UINT width;
            UINT height;
            UINT depth;
        };

        /**
       *@brief CUDA获取D3D11设备适配器
       */
        void getD3D11Adapter(IDXGIAdapter** pAdapter) noexcept(false)
        {
            Microsoft::WRL::ComPtr<IDXGIFactory> pFactory;
            HRESULT hr = CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)(pFactory.GetAddressOf()));
            if (FAILED(hr)) {
                throw std::runtime_error("No DXGI Factory created");
            }

            Microsoft::WRL::ComPtr<IDXGIAdapter> _pAdapter;
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
        /*
        *@brief d3d11 texture2d 与opencv gpumat 互转
        TODO 目前只测试了uchar4( d3d R8G8A8B8 <-> opencv CV_8UC4 ),其他暂未支持
        */
        class texture2d_cvt_gpumat
        {
        public:
            texture2d_cvt_gpumat() {};
            ~texture2d_cvt_gpumat() {
                mp_cuda_capable_adater.~ComPtr();
                mp_d3d11_device.~ComPtr();
            };

            bool init(ID3D11Device* p_d3d11_device) {
                mp_d3d11_device = p_d3d11_device;
                getD3D11Adapter(mp_cuda_capable_adater.GetAddressOf());
                if (p_d3d11_device == nullptr) {
                    HRESULT hr = windows::createD3D11Device(mp_d3d11_device.GetAddressOf(), mp_cuda_capable_adater.Get());
                    if (FAILED(hr)) {
                        return false;
                    }
                }
                return true;
            };

            cudaError_t texture2d_to_gpumat_async(ID3D11Texture2D* p_src_texture, cv::cuda::GpuMat& dst_gpumat, cudaStream_t stream = 0) {
                mt_texture_2d.p_d3d11_texture_2d = p_src_texture;

                D3D11_TEXTURE2D_DESC desc;
                mt_texture_2d.p_d3d11_texture_2d->GetDesc(&desc);

                //注册Direct3D 11资源pD3DResource以供CUDA访问
                checkCudaRet(cudaGraphicsD3D11RegisterResource(&mt_texture_2d.cuda_resource, mt_texture_2d.p_d3d11_texture_2d.Get(), cudaGraphicsRegisterFlagsNone));

                //映射图形资源以供CUDA访问
                checkCudaRet(cudaGraphicsMapResources(1, &mt_texture_2d.cuda_resource));

                //获取一个数组，通过该数组访问映射图形资源的子资源
                checkCudaRet(cudaGraphicsSubResourceGetMappedArray(&mt_texture_2d.cuda_array, mt_texture_2d.cuda_resource, 0, 0));

                checkCudaRet(cudaBindTextureToArray(mt_texture_2d.texture_ref, mt_texture_2d.cuda_array, &mt_texture_2d.cuda_array_desc));

                dst_gpumat.create(desc.Height, desc.Width, CV_MAKETYPE(CV_8U, sizeof(uchar4)));
                cv::cuda::PtrStepSz<uchar4> dst_ptr_step_sz = dst_gpumat;

                //src和dst的step可能不同
                checkCudaRet(cudaMemcpy2DFromArrayAsync(dst_ptr_step_sz.data, dst_ptr_step_sz.step,
                    mt_texture_2d.cuda_array, 0, 0, dst_ptr_step_sz.cols * sizeof(uchar4), dst_ptr_step_sz.rows, cudaMemcpyDeviceToDevice, stream));

                //保证输入输出为默认格式,D3D11默认RGBA格式,OPENCV默认BGRA格式
                cv::cuda::cvtColor(dst_gpumat, dst_gpumat, cv::COLOR_RGBA2BGRA);

                cudaUnbindTexture(mt_texture_2d.texture_ref);

                checkCudaRet(cudaGraphicsUnmapResources(1, &mt_texture_2d.cuda_resource));
                checkCudaRet(cudaGraphicsUnregisterResource(mt_texture_2d.cuda_resource));
                return cudaError::cudaSuccess;
            };

            cudaError_t gpumat_to_texture2d_async(cv::cuda::GpuMat src_gpumat, ID3D11Texture2D** pp_dst_texture, cudaStream_t stream = 0) {
                if (src_gpumat.channels() == 4) {// 默认通道类型转换
                    cv::cuda::cvtColor(src_gpumat, src_gpumat, cv::COLOR_BGRA2RGBA);
                }
                if (src_gpumat.channels() == 3) {
                    cv::cuda::cvtColor(src_gpumat, src_gpumat, cv::COLOR_BGR2RGBA);
                }

                D3D11_TEXTURE2D_DESC desc;
                windows::createTextureDesc(desc, src_gpumat.cols, src_gpumat.rows);
                mp_d3d11_device->CreateTexture2D(&desc, nullptr, mt_texture_2d.p_d3d11_texture_2d.GetAddressOf());

                //注册Direct3D 11资源以供CUDA访问
                checkCudaRet(cudaGraphicsD3D11RegisterResource(&mt_texture_2d.cuda_resource, mt_texture_2d.p_d3d11_texture_2d.Get(), cudaGraphicsRegisterFlagsNone));

                //映射图形资源以供CUDA访问
                checkCudaRet(cudaGraphicsMapResources(1, &mt_texture_2d.cuda_resource));

                //获取一个数组，通过该数组访问映射图形资源的子资源
                checkCudaRet(cudaGraphicsSubResourceGetMappedArray(&mt_texture_2d.cuda_array, mt_texture_2d.cuda_resource, 0, 0));

                checkCudaRet(cudaBindTextureToArray(mt_texture_2d.texture_ref, mt_texture_2d.cuda_array, &mt_texture_2d.cuda_array_desc));

                cv::cuda::PtrStepSz<uchar4> dst_cudamat = src_gpumat;

                checkCudaRet(cudaMemcpy2DToArrayAsync(mt_texture_2d.cuda_array, 0, 0, dst_cudamat.data, dst_cudamat.step, dst_cudamat.cols * sizeof(uchar4), dst_cudamat.rows, cudaMemcpyDeviceToDevice, stream));

                checkCudaRet(cudaUnbindTexture(mt_texture_2d.texture_ref));

                *pp_dst_texture = mt_texture_2d.p_d3d11_texture_2d.Get();
                (*pp_dst_texture)->AddRef();
                return cudaError::cudaSuccess;
            };
        private:
            Microsoft::WRL::ComPtr<ID3D11Device> mp_d3d11_device;
            Microsoft::WRL::ComPtr<IDXGIAdapter> mp_cuda_capable_adater;
            shared_texture_2d_t<uchar4> mt_texture_2d;
        };

#endif // HAVE_OPENCV
#endif // HAVE_DIRECTX
    } // namespace cuda
} // namespace common

#endif // _COMMON_CUDA_HPP_