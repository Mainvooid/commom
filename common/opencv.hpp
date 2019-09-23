/*
@brief opencv helper
@author guobao.v@gmail.com
*/
#ifndef _COMMON_OPENCV_HPP_
#define _COMMON_OPENCV_HPP_

#include <common/debuglog.hpp>
#if defined(HAVE_CUDA) && defined(HAVE_CUDA_KERNEL)
#include <common/cuda/fisheye_remap.hpp>
#endif // HAVE_CUDA && HAVE_CUDA_KERNEL

#include <opencv2/opencv.hpp>
#include <opencv2/img_hash.hpp>
#include <random>

#define OPENCV_VERSION 410
// 将参数连接并转成字符串(遇宏则展开)
#define _CV_LIB(x)  "\"" _S(x) "" _S(OPENCV_VERSION) "" ".lib\""
#define _CV_LIB_D(x)  "\"" _S(x) "" _S(OPENCV_VERSION) "" "d.lib\""

#if defined(_DEBUG) || defined(DEBUG)
#pragma comment(lib,_CV_LIB_D(opencv_img_hash))
#ifdef LINK_LIB_OPENCV_WORLD
#pragma comment(lib,_CV_LIB_D(opencv_world))
#else
#pragma comment(lib,_CV_LIB_D(opencv_core))
#pragma comment(lib,_CV_LIB_D(opencv_imgproc))
#pragma comment(lib,_CV_LIB_D(opencv_imgcodecs))
#pragma comment(lib,_CV_LIB_D(opencv_highgui))
#pragma comment(lib,_CV_LIB_D(opencv_calib3d))
#endif // LIB_OPENCV_WORLD
#ifdef HAVE_CUDA
#pragma comment(lib,_CV_LIB_D(opencv_cudawarping))
#pragma comment(lib,_CV_LIB_D(opencv_cudaimgproc))

#endif // HAVE_CUDA
#else
#pragma comment(lib,_CV_LIB(opencv_img_hash))
#ifdef LINK_LIB_OPENCV_WORLD
#pragma comment(lib,_CV_LIB(opencv_world))
#else
#pragma comment(lib,_CV_LIB(opencv_core))
#pragma comment(lib,_CV_LIB(opencv_imgproc))
#pragma comment(lib,_CV_LIB(opencv_imgcodecs))
#pragma comment(lib,_CV_LIB(opencv_highgui))
#pragma comment(lib,_CV_LIB(opencv_calib3d))
#endif // LIB_OPENCV_WORLD
#ifdef HAVE_CUDA
#pragma comment(lib,_CV_LIB(opencv_cudawarping))
#pragma comment(lib,_CV_LIB(opencv_cudaimgproc))
#endif // HAVE_CUDA
#endif // DEBUG

/**
  @addtogroup common
  @{
    @defgroup opencv opencv - opencv utilities
  @}
*/
namespace common {
    /// @addtogroup common
    /// @{
    namespace opencv {
        /// @addtogroup opencv
        /// @{
        static const std::string _TAG = "opencv";

        /**
        *@brief 由图像哈希计算图像距离
        *@return 图像哈希距离
        */
        template<typename T>
        double HashCompare(const cv::InputArray& aMat, const cv::InputArray& bMat)
        {
            cv::Mat hashA, hashB;
            std::shared_ptr<cv::img_hash::ImgHashBase> func(T::create());
            func->compute(aMat, hashA);
            func->compute(bMat, hashB);
            return func->compare(hashA, hashB);
        }

        /**
        @brief debug保存图像,路径由时间＋描述+随机数产生
        @exception std::invalid_argument 图像或路径为空
        @return 保存路径
        */
        static std::string saveDebugImage(const cv::Mat& save_image, const std::string& save_dir, const std::string& desc)
            noexcept(noexcept(!save_image.empty() && save_dir != ""))
        {
            if (save_dir == "" || save_image.empty()) {
                std::ostringstream msg;
                msg << _TAG << "..save_image or save_dir is empty";
                throw std::invalid_argument(msg.str());
            }
            time_t now_time = time(nullptr);
            char time_stamp[64];
            tm _tm;
            ::localtime_s(&_tm, &now_time);
            strftime(time_stamp, sizeof(time_stamp), "%Y-%m-%d_%H-%M-%S", &_tm);
            std::default_random_engine engine(_tm.tm_sec);
            char fname[260];
            sprintf_s(fname, "%s%s_%s_%u.png", fillDir(save_dir.data()).data(), time_stamp, desc.data(), engine());
            cv::imwrite(fname, save_image);
            return std::string(fname);
        }

        /**
        *@brief 显示指定大小的图像
        */
        static void imshowR(const std::string& img_name, cv::InputArray image, cv::Size img_size = cv::Size(960, 540))
        {
            cv::namedWindow(img_name, cv::WindowFlags::WINDOW_NORMAL);
            if (image.total() > static_cast<size_t>(img_size.height) * static_cast<size_t>(img_size.width)) {
                cv::resizeWindow(img_name, img_size);
            }
            cv::Mat _;
            if (image.isGpuMat()) {
                image.getGpuMat().download(_);
            }
            else {
                _ = image.getMat();
            }
            cv::imshow(img_name, _);
        }

        /**
        *@brief 合并俩个图像,若通道不同将合并为4通道BGRA
        */
        static cv::Mat mergeImage(cv::InputArray left, cv::InputArray right)
            noexcept(noexcept(!left.empty() && !right.empty()))
        {
            if (left.empty() || right.empty()) {
                std::ostringstream msg;
                msg << _TAG << "..InputArray is empty";
                throw std::runtime_error(msg.str());
            }
            cv::Mat _left, _right;

            if (left.isGpuMat()) {
                left.getGpuMat().download(_left);
            }
            else {
                left.getMat();
            }
            if (right.isGpuMat()) {
                right.getGpuMat().download(_right);
            }
            else {
                right.getMat();
            }

            if (left.channels() != right.channels()) {
                if (left.channels() == 1) {
                    cv::cvtColor(left, _left, cv::COLOR_GRAY2BGRA);
                }
                if (left.channels() == 3) {
                    cv::cvtColor(left, _left, cv::COLOR_BGR2BGRA);
                }
                if (right.channels() == 1) {
                    cv::cvtColor(right, _right, cv::COLOR_GRAY2BGRA);
                }
                if (right.channels() == 3) {
                    cv::cvtColor(right, _right, cv::COLOR_BGR2BGRA);
                }
            }
            cv::Mat img_merge(cv::Size(left.cols() + right.cols(), left.rows()), CV_MAKETYPE(_left.depth(), _left.channels()), cv::Scalar::all(0));
            cv::Mat dst_leftMat = img_merge(cv::Rect(0, 0, left.cols(), left.rows()));
            cv::Mat dst_rightMat = img_merge(cv::Rect(left.cols(), 0, right.cols(), right.rows()));
            _left.copyTo(dst_leftMat);
            _right.copyTo(dst_rightMat);
            return img_merge;
        }

        /**
        @brief 简单的基于棋盘定标的鱼眼相机去畸变类
        */
        class FisheyeRemap {
        public:
            FisheyeRemap() {}
            ~FisheyeRemap() {}
            /**
            @brief 计算标定参数以初始化鱼眼相机
            @param[in] chessboards 棋盘格标定板图像集合,分辨率应与相机相同
            @param[in] board_size  定标板上每行、列的内角点数(格子数-1)
            @param[in] square_size 定标板方格的宽度(mm)
            */
            void init(cv::InputArrayOfArrays chessboards, cv::Size board_size, size_t square_size) noexcept(false)
            {
                chessboards.getMatVector(m_chessboards);
                m_board_size = board_size;
                m_square_size = square_size;

                std::vector<cv::Point3f> temp_obj_point;
                for (int i = 0; i < m_board_size.height; i++) {
                    for (int j = 0; j < m_board_size.width; j++) {
                        //假设定标板放在世界坐标系中z=0的平面上
                        temp_obj_point.push_back(cv::Point3f(j * m_square_size, i * m_square_size, 0));
                    }
                }
                for (size_t i = 0; i < m_chessboards.size(); i++) {
                    cv::Mat chessboard_gray;
                    std::vector<cv::Point2f> corners;//提取的角点坐标
                    //提取角点
                    cv::findChessboardCorners(m_chessboards[i], board_size, corners,
                        cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
                    if (corners.empty()){
                        //提取不到角点坐标，检查标定图像是否满足标定要求
                        continue;
                    }
                    //else{ //DEBUG
                    //    cv::Mat _ = m_chessboards[i].clone();
                    //    cv::drawChessboardCorners(_, board_size, corners,true);
                    //    imshowR("chessboards", _);
                    //    cv::waitKey(0);
                    //}
                    cv::cvtColor(m_chessboards[i], chessboard_gray, cv::COLOR_BGR2GRAY);
                    //亚像素精确化
                    cv::cornerSubPix(chessboard_gray, corners, board_size,
                        cv::Size(-1, -1), cv::TermCriteria(3, 100, std::numeric_limits<float>::epsilon()));
                    m_corners_set.push_back(corners);
                    m_object_points_set.push_back(temp_obj_point);
                }
                //摄像机定标
                int flags = 0;
                flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;//外部优化将在每次迭代后重新计算
                flags |= cv::fisheye::CALIB_CHECK_COND;         //检查条件数的有效性
                flags |= cv::fisheye::CALIB_FIX_SKEW;           //偏斜系数alpha设置为零并保持为零,代表求解时假设内参中fx=fy

                cv::fisheye::calibrate(m_object_points_set, m_corners_set, m_chessboards[0].size(),
                    m_intrinsic_param_mat, m_distortion_coeffs, m_rotation_vectors, m_translation_vectors, flags,
                    cv::TermCriteria(3, 100, std::numeric_limits<float>::epsilon()));

                if ((m_distortion_coeffs[0] <= std::numeric_limits<float>::epsilon()
                    && m_distortion_coeffs[1] <= std::numeric_limits<float>::epsilon()
                    && m_distortion_coeffs[2] <= std::numeric_limits<float>::epsilon()
                    && m_distortion_coeffs[3] <= std::numeric_limits<float>::epsilon()
                    ) || (m_intrinsic_param_mat.rows == 0 || m_intrinsic_param_mat.cols == 0)) {
                    throw std::invalid_argument("Invalid output parameters");//可能存在某些标定图无法成功计算出参数,应排除之
                }
#if defined(HAVE_CUDA) && defined(HAVE_CUDA_KERNEL)
                m_intrinsic_param_mat_g.upload(m_intrinsic_param_mat);
                m_distortion_coeffs_g.upload(m_distortion_coeffs);
#endif // HAVE_CUDA && HAVE_CUDA_KERNEL
            }

            /**
            @brief 输入相机原始参数初始化
            @param[in] intrinsic_param_mat 摄像机内参数矩阵 3x3
            @param[in] distortion_coeffs 摄像机的4个畸变系数: k1,k2,k3,k4
            */
            void init(cv::Mat intrinsic_param_mat, cv::Vec4d distortion_coeffs)
            {
                m_intrinsic_param_mat = intrinsic_param_mat.clone();
                m_distortion_coeffs = distortion_coeffs;
#if defined(HAVE_CUDA) && defined(HAVE_CUDA_KERNEL)
                m_intrinsic_param_mat_g.upload(m_intrinsic_param_mat);
                m_distortion_coeffs_g.upload(m_distortion_coeffs);
#endif // HAVE_CUDA && HAVE_CUDA_KERNEL
            }

            /**
            @brief 应用重映射进行去畸变操作
            @param[in] src 源图像
            @param[out] dst 目标图像
            @param[in] fx 调节视场大小,乘的系数越小视场越大
            @param[in] fy 调节视场大小,乘的系数越小视场越大
            @param[in] cx 调节校正图中心,可设置为src的中心坐标x
            @param[in] cy 调节校正图中心,可设置为src的中心坐标y
            */
            void remap(cv::InputArray src, cv::OutputArray dst, double fx = 1.0, double fy = 1.0, double cx = 0, double cy = 0)
            {
                //调节视场大小, 乘的系数越小视场越大
                m_new_intrinsic_param_mat = m_intrinsic_param_mat.clone();
                m_new_intrinsic_param_mat.at<double>(0, 0) *= fx;
                m_new_intrinsic_param_mat.at<double>(1, 1) *= fy;
                //调节校正图中心，建议置于校正图中心
                if (cx != 0 && cy != 0) {
                    m_new_intrinsic_param_mat.at<double>(0, 2) = cx;
                    m_new_intrinsic_param_mat.at<double>(1, 2) = cy;
                }
                cv::fisheye::undistortImage(src, dst, m_intrinsic_param_mat, m_distortion_coeffs, m_new_intrinsic_param_mat, cv::Size());
            }

#if defined(HAVE_CUDA) && defined(HAVE_CUDA_KERNEL)
            /**@overload*/
            void remap(cv::cuda::GpuMat src, cv::cuda::GpuMat& dst, cv::cuda::Stream& stream, double fx = 1.0, double fy = 1.0, double cx = 0, double cy = 0)
            {
                //调节视场大小, 乘的系数越小视场越大
                m_new_intrinsic_param_mat = m_intrinsic_param_mat.clone();
                m_new_intrinsic_param_mat.at<double>(0, 0) *= fx;
                m_new_intrinsic_param_mat.at<double>(1, 1) *= fy;
                //调节校正图中心，建议置于校正图中心
                if (cx != 0) {
                    m_new_intrinsic_param_mat.at<double>(0, 2) = cx;
                }
                if (cy != 0) {
                    m_new_intrinsic_param_mat.at<double>(1, 2) = cy;
                }

                if (m_map1.empty() || m_map2.empty()) {
                    cuda::cuda_init_undistort_rectify_map(m_intrinsic_param_mat_g, m_distortion_coeffs_g,
                        m_new_intrinsic_param_mat, src.size(), m_map1, m_map2);
                }

                //异步调用才是线程安全的
                try {
                    cv::cuda::remap(src, dst, m_map1, m_map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(), stream);
                }
                catch (const std::exception& e) {
                    LOGD_(e.what());
                }
                stream.waitForCompletion();
            }
#endif // HAVE_CUDA && HAVE_CUDA_KERNEL

            /**
            @brief 在所有标定图上画出标定的内角点并显示
            @note 需要另外调用cv::Waitkey()来阻塞才能显示.
            */
            void show_chessboard_corners() {
                for (size_t i = 0; i < m_chessboards.size(); i++) {
                    cv::Mat _chessboards = m_chessboards[i].clone();
                    drawChessboardCorners(_chessboards, m_board_size, m_corners_set[i], true);
                    imshowR("chessboard_corners_" + convert_to_string<char>(i), _chessboards);
                }
            }
            /**
            @brief 打印相机的内参数矩阵及畸变参数
            */
            void print_camera_intrinsic_info() {
                std::stringstream ss;
                ss << "------------------------------\n"
                    << "Intrinsic parameters:\n" << m_intrinsic_param_mat << "\n"
                    << "Distortion coeffs:\n" << m_distortion_coeffs << "\n"
                    << "------------------------------\n";
                std::cout << ss.str();
                LOGI(ss.str());
            }
            /**
            @brief 计算并打印定标误差
            @return 总平均误差
            */
            double print_calibrate_errors() {
                double total_err = 0.0;
                std::vector<cv::Point2f>  new_corners;
                std::stringstream ss;
                ss << "------------------------------\n" << "Average error of each calibration image(pix):\n";
                for (int i = 0; i < m_chessboards.size(); i++)
                {
                    //通过得到的摄像机内外参数,对空间的三维点进行重投影,得到新的投影点
                    cv::fisheye::projectPoints(m_object_points_set[i], new_corners, m_rotation_vectors[i], m_translation_vectors[i], m_new_intrinsic_param_mat, m_distortion_coeffs);
                    //计算新的投影点和旧的投影点之间的误差
                    double err = cv::norm(new_corners, m_corners_set[i], cv::NormTypes::NORM_L2);
                    total_err += err /= m_board_size.width*m_board_size.height;
                    ss << i << " : " << err << "\n";
                }
                double average_total_err = total_err / m_chessboards.size();
                ss << "Total average error(pix): " << average_total_err << "\n"
                    << "------------------------------\n";
                std::cout << ss.str();
                LOGI(ss.str());
                return average_total_err;
            }
        public:
            cv::Mat m_intrinsic_param_mat;     ///<摄像机内参数矩阵
            cv::Mat m_new_intrinsic_param_mat; ///<手动调整后的摄像机内参数矩阵
            cv::Vec4d m_distortion_coeffs;     ///<摄像机的4个畸变系数: k1,k2,k3,k4
            std::vector<cv::Vec3d> m_rotation_vectors;   ///<每幅图像的旋转向量
            std::vector<cv::Vec3d> m_translation_vectors;///<每幅图像的平移向量
        private:
#if defined(HAVE_CUDA) && defined(HAVE_CUDA_KERNEL)
            cv::cuda::GpuMat m_intrinsic_param_mat_g;///<摄像机内参数矩阵
            cv::cuda::GpuMat m_distortion_coeffs_g;  ///<摄像机的4个畸变系数: k1,k2,k3,k4
            cv::cuda::GpuMat m_map1, m_map2;         ///<畸变校正矩阵
#endif // HAVE_CUDA && HAVE_CUDA_KERNEL
            std::vector<cv::Mat> m_chessboards;///<棋盘格
            cv::Size m_board_size;///<定标板上每行、列的内角点数,等于格子数-1
            size_t m_square_size; ///<定标板上格子的宽度mm
            std::vector<std::vector<cv::Point3f>> m_object_points_set;///<保存定标板上角点的三维坐标,一个vector对应一张定标板
            std::vector<std::vector<cv::Point2f>> m_corners_set;      ///<标定图的角点坐标
        };
        /// @}
    } // namespace opencv
    /// @}
} // namespace common

#endif // _COMMON_OPENCV_HPP_