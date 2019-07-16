#include <common/cuda/texture_reference.hpp>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <channel_descriptor.h>
#include <type_traits>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_texture_types.h>

namespace common {
    namespace cuda {

        //纹理参照系必须定义在所有函数体外(全局性),需要显式声明,用NVCC编译,不支持3元组
        texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef_2d_uchar4;
        texture<float4, cudaTextureType2D, cudaReadModeElementType> texRef_2d_float4;

        cudaError_t cuda_get_texture_reference_2d_uchar4(const textureReference ** texref) {
            return cudaGetTextureReference(texref, &texRef_2d_uchar4);
        }
        cudaError_t cuda_get_texture_reference_2d_float4(const textureReference ** texref) {
            return cudaGetTextureReference(texref, &texRef_2d_float4);
        }
    }// namespace cuda
} // namespace common