/*
@brief usage sample for opencv.hpp.
@author guobao.v@gmail.com
*/

#include <common/opencv.hpp>

using namespace common::opencv;

int main()
{
    cv::UMat left(cv::Size(1280, 720), CV_8UC3, cv::Scalar::all(40));
    cv::UMat right(cv::Size(960, 540), CV_8UC3, cv::Scalar::all(100));
    cv::Mat result;

    double PHashCompareResult = HashCompare<cv::img_hash::PHash>(left, right);                    //感知hash
    double ColorMomentHashCompareResult = HashCompare<cv::img_hash::ColorMomentHash>(left, right);//颜色矩hash
    double AverageHashCompareResult = HashCompare<cv::img_hash::AverageHash>(left, right);        //均值hash
    double BlockMeanHashCompareResult = HashCompare<cv::img_hash::BlockMeanHash>(left, right);    //块均值hash

    result = mergeImage(left, right);
    saveDebugImage(result, ".", "result");
    imshowR("resultR", result);
    cv::waitKey(0);
}