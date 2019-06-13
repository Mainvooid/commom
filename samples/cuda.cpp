/*
@brief usage sample for cuda.hpp
@author guobao.v@gmail.com
*/

#define HAVE_OPENCL
#define HAVE_OPENCV
#define HAVE_DIRECTX
#define HAVE_CUDA 
#define HAVE_CUDA_DEVICE 

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
shared_texture_2d_t<uchar4> g_texture_2d;

void test_texture2d_to_gpumat(texture2d_cvt_gpumat& cvt) {
    loadTextureFromFile(g_pd3dDevice.Get(), g_texture_2d.p_d3d11_texture_2d.GetAddressOf(), L"samples/data/1920_1080.jpg");

    cv::cuda::GpuMat dst;
    cvt.texture2d_to_gpumat(g_texture_2d.p_d3d11_texture_2d.Get(), dst);//OK
    
    cv::Mat dstmat;
    dst.download(dstmat);
    imshowR("texture2d_to_gpumat", dstmat);
    cv::waitKey(2000);
    dst.release();
}

void test_gpumat_to_texture2d(texture2d_cvt_gpumat& cvt){
cv::Mat src = cv::imread("samples/data/1920_1080.jpg");
cv::cuda::GpuMat srcGpuMat;
srcGpuMat.upload(src);
cvt.gpumat_to_texture2d(srcGpuMat, g_texture_2d.p_d3d11_texture_2d.GetAddressOf(), g_pd3dDevice.Get());

saveTextureToFile(g_pd3dDevice.Get(), g_texture_2d.p_d3d11_texture_2d.Get(), L"dst.png");
cv::Mat dstmat2 = cv::imread("dst.png");
imshowR("gpumat_to_texture2d", dstmat2);
cv::waitKey(2000);
remove("dst.png");
srcGpuMat.release();
}

int main() {
    checkCudaRet(cuda_get_texture_reference<uchar4>(&g_texture_2d.texture_ref));

    if (checkCUDADevice() != cudaSuccess)return -1;
    createD3D11Device(g_pd3dDevice.GetAddressOf());
    texture2d_cvt_gpumat cvt;
    cvt.init();

    std::thread run([&]{
        test_texture2d_to_gpumat(cvt);// OK
    });

    std::thread run1([&] {
        test_gpumat_to_texture2d(cvt);// OK
    });
    run.join();
    run1.join();
}