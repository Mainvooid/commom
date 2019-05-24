/*
@brief usage sample for debuglog.hpp.
@author guobao.v@gmail.com
*/

#include <common/debuglog.hpp>
#include <iostream>

using namespace common;
using namespace common::debuglog;

void func()
{
    logger<> log(_T("TEST"), level_e::Trace);
    std::wstring wstr = L"Test测试123";
    std::string str = "Test测试123";

#if defined(_UNICODE) or defined(UNICODE)
    log.Log(wstr, level_e::Trace);
    log.Log(wstr, level_e::Debug, __FUNCTION__, __FILE__, __LINE__);
    LOGI(wstr);
    LOGW(wstr.data());
    LOGE(wstr, __FUNCTION__, __FILE__, __LINE__);
    LOGF(wstr.data(), __FUNCTION__, __FILE__, __LINE__);
    LOGE_(wstr);
#else
    log.Log(str, level_e::Trace);
    log.Log(str, level_e::Debug, __FUNCTION__, __FILE__, __LINE__);
    LOGI(str);
    LOGW(str.data());
    LOGE(str, __FUNCTION__, __FILE__, __LINE__);
    LOGF(str.data(), __FUNCTION__, __FILE__, __LINE__);
    LOGE_(str);
#endif
}
int main()
{
    func();
}
/*
Trace [TEST] : Test测试123
Debug [TEST] : Test测试123   [...\common\samples\debuglog.cpp(37) func]
Info  [G] : Test测试123
Warn  [G] : Test测试123
Error [G] : Test测试123   [...\common\samples\debuglog.cpp(40) func]
Fatal [G] : Test测试123   [...\common\samples\debuglog.cpp(41) func]

Trace [TEST] : Test测试123
Debug [TEST] : Test测试123   [...\common\samples\debuglog.cpp(44) func]
Info  [G] : Test测试123
Warn  [G] : Test测试123
Error [G] : Test测试123   [...\common\samples\debuglog.cpp(47) func]
Fatal [G] : Test测试123   [...\common\samples\debuglog.cpp(48) func]
*/
