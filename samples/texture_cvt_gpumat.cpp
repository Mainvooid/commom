/*
@brief usage sample for cuda.hpp / class texture2d_cvt_gpumat
@author guobao.v@gmail.com
*/

#define HAVE_OPENCL
#define HAVE_OPENCV
#define HAVE_DIRECTX
#define HAVE_CUDA 
#define HAVE_CUDA_KERNEL 

#include <common/cuda.hpp>
#include <common/opencv.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <thread>
using namespace common;
using namespace common::cuda;
using namespace common::opencv;
using namespace common::windows;
using namespace Microsoft::WRL;

ComPtr<IDXGIAdapter> g_pCudaCapableAdapter;
ComPtr<ID3D11Device> g_pd3dDevice;
ComPtr<ID3D11DeviceContext> g_pd3dDeviceContext;
shared_texture_2d_t<uchar4, cudaReadModeElementType> g_texture_2d;

void test_texture2d_to_gpumat(texture2d_cvt_gpumat<>& cvt) {
    HRESULT hr = loadTextureFromFile(g_pd3dDevice.Get(), g_texture_2d.p_d3d11_texture_2d.GetAddressOf(), L"samples/data/1920_1080.jpg",
        DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED);
    if (FAILED(hr)) { LOGE_("loadTextureFromFile"); }
    cv::cuda::GpuMat dst;
    cvt.bind_for_texture2d_to_gpumat(g_texture_2d.p_d3d11_texture_2d.Get());
    std::function<void(void)> f1 = [&]() {
        cvt.texture2d_to_gpumat(dst);//OK
    };
    std::cout << "texture2d_to_gpumat CPU时间:" << getFnDuration(f1) << "ms" << std::endl;//1ms
    cvt.unbind_for_texture2d_to_gpumat();

    cv::Mat dstmat;
    cv::cuda::cvtColor(dst, dst, cv::COLOR_RGBA2BGRA);
    dst.download(dstmat);
    imshowR("texture2d_to_gpumat", dstmat);
    cv::waitKey(2000);
    dst.release();
}

void test_gpumat_to_texture2d(texture2d_cvt_gpumat<>& cvt) {
    cv::Mat src = cv::imread("samples/data/1920_1080.jpg");
    cv::cuda::GpuMat srcGpuMat;
    srcGpuMat.upload(src);
    cvt.bind_for_gpumat_to_texture2d(1920, 1080);
    std::function<void(void)> f1 = [&]() {
        cvt.gpumat_to_texture2d(srcGpuMat, g_texture_2d.p_d3d11_texture_2d.GetAddressOf(), g_pd3dDevice.Get());
    };
    std::cout << "gpumat_to_texture2d CPU时间:" << getFnDuration(f1) << "ms" << std::endl;//40ms
    cvt.unbind_for_gpumat_to_texture2d();
    saveTextureToFile(g_pd3dDevice.Get(), g_texture_2d.p_d3d11_texture_2d.Get(), L"dst.png");
    cv::Mat dstmat2 = cv::imread("dst.png");
    imshowR("gpumat_to_texture2d", dstmat2);
    cv::waitKey(2000);
    remove("dst.png");
    srcGpuMat.release();
}

int main() {
    cuda_get_texture_reference<uchar4, cudaTextureType2D, cudaReadModeElementType>()(&g_texture_2d.texture_ref);

    if (checkCUDADevice() != cudaSuccess) return -1;
    createD3D11Device(g_pd3dDevice.GetAddressOf());
    texture2d_cvt_gpumat<uchar4> cvt;

    std::thread run([&] {
        test_texture2d_to_gpumat(cvt);// OK
    });

    std::thread run1([&] {
        test_gpumat_to_texture2d(cvt);// OK
    });
    run.join();
    run1.join();
}
