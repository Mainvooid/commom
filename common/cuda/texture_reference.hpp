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
        cudaError_t cuda_get_texture_reference_2d_uchar4(const textureReference ** texref);
        cudaError_t cuda_get_texture_reference_2d_float4(const textureReference ** texref);

        /**
        @brief 获取CUDA纹理参考系对象
        @note 暂只支持2d uchar4,float4 纹理
        */
        template<typename T, int texType = cudaTextureType2D, enum cudaTextureReadMode mode = cudaReadModeElementType>
        struct cuda_get_texture_reference;
        /**@overload*/
        template<>
        struct cuda_get_texture_reference<uchar4, cudaTextureType2D, cudaReadModeElementType> {
            cudaError_t operator()(const textureReference ** texref) {
                return cuda_get_texture_reference_2d_uchar4(texref);
            }
        };
        /**@overload*/
        template<>
        struct cuda_get_texture_reference<float4, cudaTextureType2D, cudaReadModeElementType> {
            cudaError_t operator()(const textureReference ** texref) {
                return cuda_get_texture_reference_2d_float4(texref);
            }
        };
        /// @}
    }// namespace cuda
    /// @}
} // namespace common
#endif //_COMMON_CUDA_TEXTURE_REFERENCE_CUH_