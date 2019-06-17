/*
@brief opencl helper
@author guobao.v@gmail.com
*/
#ifndef _COMMON_OPENCL_HPP_
#define _COMMON_OPENCL_HPP_

#include <common/debuglog.hpp>
#include <CL/cl.hpp>
#pragma comment(lib,"opencl.lib")

/**
  @addtogroup common
  @{
    @defgroup opencl opencl - opencl utilities
  @}
*/
namespace common {
    namespace opencl {
        /// @addtogroup opencl
        /// @{
        /*
        *@brief 打印opencl设备信息
        */
        void printPlatformInfo()
        {
            cl_platform_id *platform;
            cl_uint num_platform;
            cl_int err;

            err = clGetPlatformIDs(0, NULL, &num_platform);
            platform = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platform);

            err = clGetPlatformIDs(num_platform, platform, NULL);

            size_t size;
            std::wstringstream ss;
            for (cl_uint i = 0; i < num_platform; i++)
            {
                // name
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, 0, NULL, &size);
                char *name = (char *)malloc(size);
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME, size, name, NULL);
                ss << "CL_PLATFORM_NAME: " << name;
                LOGI(ss.str());
                ss.str(L"");

                // vendor
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
                char *vendor = (char *)malloc(size);
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_VENDOR, size, vendor, NULL);
                ss << "CL_PLATFORM_VENDOR: " << vendor;
                LOGI(ss.str());
                ss.str(L"");

                // version
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, 0, NULL, &size);
                char *version = (char *)malloc(size);
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION, size, version, NULL);
                ss << "CL_PLATFORM_VERSION: " << version;
                LOGI(ss.str());
                ss.str(L"");

                // profile
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, 0, NULL, &size);
                char *profile = (char *)malloc(size);
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE, size, profile, NULL);
                ss << "CL_PLATFORM_PROFILE: " << profile;
                LOGI(ss.str());
                ss.str(L"");

                // extensions
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, 0, NULL, &size);
                char *extensions = (char *)malloc(size);
                err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS, size, extensions, NULL);
                ss << "CL_PLATFORM_EXTENSIONS: " << extensions;
                LOGI(ss.str());
                ss.str(L"");

                if (i < num_platform - 1) {
                    LOGI(L"");
                }
                free(name);
                free(vendor);
                free(version);
                free(profile);
                free(extensions);
            }
        }
        /// @}
    } // namespace opencl
} // namespace common
#endif // _COMMON_OPENCL_HPP_