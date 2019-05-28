/*
@brief cuda common, cuda_d3d11_interop
@author guobao.v@gmail.com
*/
#ifdef HAVE_CUDA

#ifndef _COMMON_CUDA_HPP_
#define _COMMON_CUDA_HPP_

#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>
#include <helper_cuda.h>
#include <helper_functions.h> 

#pragma comment(lib,"cudart.lib")

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
#endif // HAVE_DIRECTX

namespace common {
    namespace cuda {

        /**
        *@brief CUDA设备检查
        */
        cudaError_t checkCUDADevice()
        {
            int deviceCount = 0;
            checkCudaErrors(cudaGetDeviceCount(&deviceCount));
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
        struct shared_texture_t
        {
            Microsoft::WRL::ComPtr<ID3D11Texture2D>          pTexture;
            Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> pSRView;
            cudaGraphicsResource* cudaResource;
            void*                 cudaLinearMemory;
            size_t                pitch;
        };
        struct shared_texture_cube_t : shared_texture_t
        {
            int                  size;
        };
        struct shared_texture_2d_t : shared_texture_t
        {
            int                   width;
            int                   height;
        };
        struct shared_texture_3d_t : shared_texture_t
        {
            int                  width;
            int                  height;
            int                  depth;
        };

        /**
       *@brief CUDA获取D3D11设备适配器
       */
        void getD3D11Adapter(IDXGIAdapter* pAdapter) noexcept(false)
        {
            Microsoft::WRL::ComPtr<IDXGIFactory> pFactory;
            HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory), (void**)(pFactory.GetAddressOf()));
            if (FAILED(hr)) {
                throw std::runtime_error("No DXGI Factory created");
            }

            Microsoft::WRL::ComPtr<IDXGIAdapter> _pAdapter;
            for (UINT i = 0; !pAdapter; ++i)
            {
                //获取一个候选DXGI适配器
                hr = pFactory->EnumAdapters(i, _pAdapter.ReleaseAndGetAddressOf());
                if (FAILED(hr)) { break; }

                //查询是否存在相应的计算设备
                int cuDevice;
                checkCudaErrors(cudaD3D11GetDevice(&cuDevice, _pAdapter.Get()));
                pAdapter = _pAdapter.Get();
                //pAdapter->AddRef();
            }
            if (!pAdapter) {
                throw std::runtime_error("No IDXGIAdapter");
            }
        }
#endif // HAVE_DIRECTX
    }
} // namespace common

#endif // _COMMON_CUDA_HPP_
#endif // HAVE_CUDA