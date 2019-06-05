#include <common/cuda/texture_reference.cuh>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include <channel_descriptor.h>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <cuda_texture_types.h>

namespace common {
    namespace cuda {

        //纹理参照系必须定义在所有函数体外,需要显式声明,用NVCC编译,不支持3元组
        texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef_2d_uchar4;

        template<class T, int texType, enum cudaTextureReadMode mode >
        cudaError_t cuda_get_texture_reference(const textureReference ** texref)
        {
            //TODO 需要根据texType进行对象选择(texture需要显式定义,若为模板,在release模式下会初始化失败),暂只支持2d_uchar4
            return cudaGetTextureReference(texref, &texRef_2d_uchar4);
        }

        cudaError_t cuda_get_texture_reference_2d_uchar4(const textureReference ** texref)
        {
            return cuda_get_texture_reference<uchar4, cudaTextureType2D, cudaReadModeElementType>(texref);
        }

    }// namespace cuda
} // namespace common