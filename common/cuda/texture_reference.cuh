#ifndef _COMMON_CUDA_TEXTURE_REFERENCE_CUH_
#define _COMMON_CUDA_TEXTURE_REFERENCE_CUH_

#include <texture_types.h>
#include <driver_types.h>

namespace common {
    namespace cuda {
        template<class T, int texType = cudaTextureType2D, enum cudaTextureReadMode mode = cudaReadModeElementType>
        cudaError_t cuda_get_texture_reference(const textureReference ** texref);

        cudaError_t cuda_get_texture_reference_2d_uchar4(const textureReference ** texref);

    }// namespace cuda
} // namespace common
#endif //_COMMON_CUDA_TEXTURE_REFERENCE_CUH_