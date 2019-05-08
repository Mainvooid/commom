/*
@brief opencv helper
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_OPENCV_HPP_
#define _COMMON_OPENCV_HPP_

#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include <random>

#pragma comment(lib,"opencv_world401.lib")
#pragma comment(lib,"opencv_img_hash401.lib")

namespace common {

    namespace opencv {
        static const std::string _TAG = "opencv";

        /**
        *@brief 由图像哈希计算图像距离
        *@return 图像哈希距离
        */
        template<typename T>
        double HashCompare(const cv::InputArray & aMat, const cv::InputArray & bMat)
        {
            cv::Mat hashA, hashB;
            std::shared_ptr<cv::img_hash::ImgHashBase> func(T::create());
            func->compute(aMat, hashA);
            func->compute(bMat, hashB);
            return func->compare(hashA, hashB);
        }

        /**
        *@brief debug保存图像,路径由时间＋描述+随机数产生
        *@return 是否保存成功
        */
        static bool saveDebugImage(const cv::Mat& save_image, const std::string& save_dir, const std::string& desc)
        {
            if (save_dir == "" || save_image.empty()) { return false; }
            time_t now_time = time(nullptr);
            char tmp[64];
            tm _tm;
            ::localtime_s(&_tm, &now_time);
            std::strftime(tmp, sizeof(tmp), "%Y-%m-%d_%H-%M-%S", &_tm);
            std::default_random_engine engine(_tm.tm_sec);
            char fname[260];
            sprintf_s(fname, "%s\\%s_%s_%d.png", save_dir.data(), tmp, desc.data(), static_cast<size_t>(engine()));
            cv::imwrite(fname, save_image);
            return true;
        }

        /**
        *@brief 显示指定大小的图像
        */
        static void imshowR(std::string img_name, const cv::InputArray& image, cv::Size img_size = cv::Size(960, 540))
        {
            cv::namedWindow(img_name, cv::WindowFlags::WINDOW_NORMAL);
            if (image.total() > static_cast<size_t>(img_size.height*img_size.width)) {
                cv::resizeWindow(img_name, img_size);
            }
            cv::imshow(img_name, image.getMat());
        }

        /**
        *@brief 合并俩个图像
        */
        static cv::Mat mergeImage(const cv::InputArray & left, const cv::InputArray & right)
            noexcept(noexcept(!left.empty() || !right.empty()))
        {
            if (left.empty() || right.empty()) {
                std::ostringstream msg;
                msg << _TAG << "..InputArray is empty";
                throw std::invalid_argument(msg.str());
            }
            cv::Mat img_merge(cv::Size(left.cols() + right.cols(), left.rows()), CV_MAKETYPE(left.depth(), 3), cv::Scalar::all(0));
            cv::Mat dst_leftMat = img_merge(cv::Rect(0, 0, left.cols(), left.rows()));
            cv::Mat dst_rightMat = img_merge(cv::Rect(left.cols(), 0, right.cols(), right.rows()));
            left.getMat().copyTo(dst_leftMat);
            right.getMat().copyTo(dst_rightMat);
            return img_merge;
        }

    } // namespace opencv

} // namespace common

#endif // _COMMON_OPENCV_HPP_
