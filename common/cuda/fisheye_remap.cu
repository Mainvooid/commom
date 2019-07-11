#include <common/cuda/fisheye_remap.hpp>
#ifndef __CUDACC__
#define __CUDACC__
#endif

#include <device_functions.h>
#include <device_launch_parameters.h>

namespace common {
    namespace cuda {
        namespace kernel {
            __global__ void cuda_init_undistort_rectify_map_kernel(cv::cuda::PtrStepSz<double> K, cv::cuda::PtrStepSz<double> D,
                cv::cuda::PtrStepSz<double> iR, cv::Size size, cv::cuda::PtrStepSz<float> map1, cv::cuda::PtrStepSz<float> map2)
            {
                int i = blockDim.x*blockIdx.x + threadIdx.x;//一个线程处理一行
                double f[2], c[2];
                f[0] = K(0, 0);//fx
                f[1] = K(1, 1);//fy
                c[0] = K(0, 2);//cx
                c[1] = K(1, 2);//cy
                if (i < size.height) {
                    float* m1f = map1.ptr(i);
                    float* m2f = map2.ptr(i);
                    double _x = i * iR(0, 1) + iR(0, 2);
                    double _y = i * iR(1, 1) + iR(1, 2);
                    double _w = i * iR(2, 1) + iR(2, 2);
                    for (int j = 0; j < size.width; ++j) {
                        double u, v;
                        if (_w <= 0) {
                            u = (_x > 0) ? -DBL_MAX : DBL_MAX;
                            v = (_y > 0) ? -DBL_MAX : DBL_MAX;
                        }
                        else {
                            double x = _x / _w, y = _y / _w;
                            double r = sqrt(x*x + y * y);
                            double theta = atan(r);
                            double theta2 = theta * theta, theta4 = theta2 * theta2, theta6 = theta4 * theta2, theta8 = theta4 * theta4;
                            double theta_d = theta * (1 + D[0] * theta2 + D[1] * theta4 + D[2] * theta6 + D[3] * theta8);
                            double scale = (r == 0) ? 1.0 : theta_d / r;
                            u = f[0] * x*scale + c[0];
                            v = f[1] * y*scale + c[1];
                        }
                        m1f[j] = (float)u;//CV_32FC1
                        m2f[j] = (float)v;
                        _x += iR(0, 0);
                        _y += iR(1, 0);
                        _w += iR(2, 0);
                    }
                }
            }
        } // namespace kernel

        cudaError_t cuda_init_undistort_rectify_map(cv::cuda::GpuMat& K, cv::cuda::GpuMat& D, const cv::Mat& Knew,
            const cv::Size& size, cv::cuda::GpuMat& map1, cv::cuda::GpuMat& map2)
        {
            map1.create(size, CV_32FC1);
            map2.create(size, CV_32F);
            cv::Matx33d _Knew;
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    _Knew(i, j) = Knew.at<double>(i, j);
                }
            }
            cv::Matx33d iR = (_Knew * cv::Matx33d::eye()).inv(cv::DECOMP_SVD);//奇异值分解
            cv::cuda::GpuMat iR_g(iR);
            //10个block每个108个线程,每个线程处理一行
            kernel::cuda_init_undistort_rectify_map_kernel << <108, 10, 0 >> > (K, D, iR_g, size, map1, map2);
            return cudaGetLastError();
        }
    } // namespace cuda
} // namespace common