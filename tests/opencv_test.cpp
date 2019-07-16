/*
@brief unit test for opencv.hpp
@author guobao.v@gmail.com
*/
#include "gtest/gtest.h"
#include <common/opencv.hpp>
using namespace common::opencv;

TEST(opencv, img_hash) {
    cv::UMat left(cv::Size(10, 10), CV_8UC3, cv::Scalar::all(40));
    cv::UMat right(cv::Size(10, 10), CV_8UC3, cv::Scalar::all(100));
    double PHashCompareResult = HashCompare<cv::img_hash::PHash>(left, right);                    //感知hash
    double ColorMomentHashCompareResult = HashCompare<cv::img_hash::ColorMomentHash>(left, right);//颜色矩hash
    double AverageHashCompareResult = HashCompare<cv::img_hash::AverageHash>(left, right);        //均值hash
    double BlockMeanHashCompareResult = HashCompare<cv::img_hash::BlockMeanHash>(left, right);    //块均值hash
    EXPECT_EQ(PHashCompareResult,0);
    EXPECT_NE(ColorMomentHashCompareResult, 0);
    EXPECT_EQ(AverageHashCompareResult, 0);
    EXPECT_EQ(BlockMeanHashCompareResult, 0);
}
TEST(opencv, mergeImage) {
    cv::UMat left(cv::Size(20, 20), CV_8UC3, cv::Scalar::all(40));
    cv::UMat right(cv::Size(10, 10), CV_8UC3, cv::Scalar::all(100));
    cv::Mat result;
    result = mergeImage(left, right);
    EXPECT_NE(result.cols, 0);
    EXPECT_NE(result.rows, 0);
}
TEST(opencv, saveDebugImage) {
    cv::Mat result(cv::Size(960, 540), CV_8UC3, cv::Scalar::all(60));
    std::string fname= saveDebugImage(result, ".", "result");
    EXPECT_NE(fname, "");
    remove(fname.data());
}