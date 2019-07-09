/*
@brief usage sample for opencv.hpp / class FisheyeRemap
@author guobao.v@gmail.com
*/

#include <common/opencv.hpp>
using namespace common;

std::vector<cv::Mat> read_chessboards(size_t num = 8) {
    std::vector<cv::Mat> chessboard_all;
    for (size_t i = 0; i < num; i++) {
        std::string path = "samples/data/chessboard/cb_" + convert_to_string<char>(i) + ".jpg";
        cv::Mat cb = cv::imread(path);
        chessboard_all.push_back(cb);
    }
    return chessboard_all;
}

int main() {
    std::vector<cv::Mat> chessboard_all = read_chessboards();
    cv::Size board_size = cv::Size(9, 6);
    size_t square_size = 25;
    opencv::FisheyeRemap FR;
    FR.init(chessboard_all, board_size, square_size);

    FR.show_chessboard_corners();

    for (size_t i = 0; i < chessboard_all.size(); i++)
    {
        cv::Mat result;
        FR.remap(chessboard_all[i], result);//视场较小,误差较小
        opencv::imshowR("merge0_" + convert_to_string<char>(i), opencv::mergeImage(chessboard_all[i], result), cv::Size(720 * 2, 405));
    }
    FR.print_camera_intrinsic_info();
    FR.print_calibrate_errors();

    std::cout << std::endl;
    for (size_t i = 0; i < chessboard_all.size(); i++)
    {
        cv::Mat result;
        FR.remap(chessboard_all[i], result, 0.5, 0.5);//扩大视场,但可能增大误差
        opencv::imshowR("merge1_" + convert_to_string<char>(i), opencv::mergeImage(chessboard_all[i], result), cv::Size(720 * 2, 405));
    }
    FR.print_camera_intrinsic_info();
    FR.print_calibrate_errors();
    cv::waitKey(0);
};