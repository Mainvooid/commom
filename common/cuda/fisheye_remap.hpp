#ifndef _COMMON_CUDA_FISHEYE_REMAP_CUH_
#define _COMMON_CUDA_FISHEYE_REMAP_CUH_

#include <cuda_runtime.h>
#include <driver_types.h>
#include <opencv2/core/cuda.hpp> 

namespace common {
    /// @addtogroup common
    /// @{
    namespace cuda {
        /// @addtogroup cuda
        /// @{
        namespace kernel {
            /// @addtogroup kernel
            /// @{
        /**
        @see cuda_init_undistort_rectify_map
        */
            __global__ void cuda_init_undistort_rectify_map_kernel(cv::cuda::PtrStepSz<double> K, cv::cuda::PtrStepSz<double> D,
                cv::cuda::PtrStepSz<double> iR,cv::Size size, cv::cuda::PtrStepSz<float> map1, cv::cuda::PtrStepSz<float> map2);
            /// @}
        } // namespace kernel

        /**
        @brief cv::fisheye::initUndistortRectifyMap cuda版本
        通过cv::cuda::remap()计算图像变换的不失真和校正图.如果D为空,则使用零失真,如果R或P为空,则使用单位矩阵.
        @param[in] K 相机矩阵(3x3)
        @param[in] D 畸变系数 (k1,k2,k3,k4).
        @param[in] Knew 新的相机矩阵(3x3)
        @param[in] size 未扭曲的图像大小
        @param[out] map1 第一个输出图
        @param[out] map2 第二个输出图
        */
        cudaError_t cuda_init_undistort_rectify_map(cv::cuda::GpuMat& K, cv::cuda::GpuMat& D, const cv::Mat& Knew,
            const cv::Size& size, cv::cuda::GpuMat& map1, cv::cuda::GpuMat& map2);
        /// @}
    } // namespace cuda
    /// @}
} // namespace common
#endif // _COMMON_CUDA_FISHEYE_REMAP_CUH_