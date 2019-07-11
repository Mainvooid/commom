#ifndef _COMMON_CUDA_TEXTURE_REFERENCE_CUH_
#define _COMMON_CUDA_TEXTURE_REFERENCE_CUH_

#include <texture_types.h>
#include <driver_types.h>
namespace common {
    /// @addtogroup common
    /// @{
    namespace cuda {
        /// @addtogroup cuda
        /// @{
        /**
        @brief 获取CUDA纹理参考系对象
        @note 暂只支持2d uchar4 纹理
        */
        template<class T, int texType = cudaTextureType2D, enum cudaTextureReadMode mode = cudaReadModeElementType>
        cudaError_t cuda_get_texture_reference(const textureReference ** texref);

        /**
        @brief [特化]获取CUDA 2D uchar4 类型的纹理参考系对象
        @param[out] texref 纹理参考系对象
        */
        cudaError_t cuda_get_texture_reference_2d_uchar4(const textureReference ** texref);
        /// @}
    }// namespace cuda
    /// @}
} // namespace common
#endif //_COMMON_CUDA_TEXTURE_REFERENCE_CUH_