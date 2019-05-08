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
    logger log(_T("test"), level_e::Info);
    std::wstring wstr = _T("Test测试123");
    std::string str = "Test测试123";

    log.Log(str, level_e::Info);
    log.Log(wstr, level_e::Info);
    log.Log(str, level_e::Info, __FUNCTION__, __FILE__, __LINE__);
    log.Log(wstr, level_e::Info, __FUNCTION__, __FILE__, __LINE__);

    LOGT(wstr);
    LOGI(wstr.data());
    LOGD(str);
    LOGW(str.data());
    LOGE(str, __FUNCTION__, __FILE__, __LINE__);
    LOGF(str.data(), __FUNCTION__, __FILE__, __LINE__);
    LOGD(wstr, __FUNCTION__, __FILE__, __LINE__);
    LOGD(wstr.data(), __FUNCTION__, __FILE__, __LINE__);
}
int main()
{
    func();
}
/*
Info  [test] : Test测试123
Info  [test] : Test测试123
Info  [test] : Test测试123   [...\common\samples\debuglog.cpp(20) func]
Info  [test] : Test测试123   [...\common\samples\debuglog.cpp(21) func]
Trace [G] : Test测试123
Info  [G] : Test测试123
Debug [G] : Test测试123
Warn  [G] : Test测试123
Error [G] : Test测试123   [...\common\samples\debuglog.cpp(27) func]
Fatal [G] : Test测试123   [...\common\samples\debuglog.cpp(28) func]
Debug [G] : Test测试123   [...\common\samples\debuglog.cpp(29) func]
Debug [G] : Test测试123   [...\common\samples\debuglog.cpp(30) func]
*/
