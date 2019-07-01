/*
@brief unit test for debuglog.hpp
@author guobao.v@gmail.com
*/
#include "gtest/gtest.h"
#include <common/debuglog.hpp>
using namespace common;
using namespace common::debuglog;

TEST(debuglog, OutputDebugStringEx) {
    OutputDebugStringA("EQ(");
    OutputDebugStringEx()("%s", "Test测试123");
    OutputDebugStringA(", Test测试123)\n");

    OutputDebugStringA("EQ(");
    OutputDebugStringEx()(L"%s", L"Test测试123");
    OutputDebugStringA(", Test测试123)\n");
}
TEST(debuglog, logger__constructor) {
    logger log1("name1", level_e::Trace);
    logger log2(L"name2", level_e::Trace);
    EXPECT_EQ(log1.getName(), "name1");
    EXPECT_EQ(log1.getLevel(), level_e::Trace);
    log1.setName("name11");
    EXPECT_EQ(log1.getName(), "name11");
    log1.setName(L"name11");
    EXPECT_EQ(log1.getName(), "name11");
    EXPECT_EQ(log1.getWName(), L"name11");
    log1.setLevel(level_e::Error);
    EXPECT_EQ(log1.getLevel(), level_e::Error);
}
TEST(debuglog, logger__Log) {
    logger log("name", level_e::Trace);
    std::wstring wstr = L"Test测试123";
    std::string str = "Test测试123";

    OutputDebugStringA("----------以下15条测试应正常打印----------\n");
    log.Log(wstr, level_e::Trace);
    log.Log(wstr, level_e::Debug, __FUNCTION__, __FILE__, __LINE__);
    LOGI("Test测试123");
    LOGI(wstr);
    LOGW(wstr.data());
    LOGE(wstr, __FUNCTION__, __FILE__, __LINE__);
    LOGF(wstr.data(), __FUNCTION__, __FILE__, __LINE__);
    LOGE_(wstr);

    log.Log(str, level_e::Trace);
    log.Log(str, level_e::Debug, __FUNCTION__, __FILE__, __LINE__);
    LOGI(str);
    LOGW(str.data());
    LOGE(str, __FUNCTION__, __FILE__, __LINE__);
    LOGF(str.data(), __FUNCTION__, __FILE__, __LINE__);
    LOGE_(str);
    OutputDebugStringA("--------------------\n");
}
