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
    log.printLog(str, level_e::Info);
    log.printLog(wstr.data(), level_e::Info);
    log.printLog(str, level_e::Info, __FUNCTION__, __FILE__, __LINE__);
    log.printLog(wstr.data(), level_e::Info, __FUNCTION__, __FILE__, __LINE__);
}
int main()
{
    func();
}
/*
Info [test] : Test测试123
Info [test] : Test测试123
Info [test] : Test测试123   [...\common\samples\debuglog.cpp(19) func]
Info [test] : Test测试123   [...\common\samples\debuglog.cpp(20) func]
*/
