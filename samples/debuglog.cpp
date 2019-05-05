/*
@brief usage sample for debuglog.h.
@author guobao.v@gmail.com
*/

#include <common/debuglog.h>
#include <iostream>

using namespace common;
using namespace common::debuglog;

void test()
{
    logger log("test", level_e::Info);
    std::wstring wstr = _T("Test测试123");
    std::string str = "Test测试123";
    log.printLog(str, level_e::Info);
    log.printLog(wstr.data(), level_e::Info);
    log.printLog(str, level_e::Info, __FUNCTION__, __FILE__, __LINE__);
    log.printLog(wstr.data(), level_e::Info, __FUNCTION__, __FILE__, __LINE__);
}
int main()
{
    test();
}
/*
Info : Test测试123
Info : Test测试123
Info : Test测试123   [...\common\samples\debuglog.cpp(19) test]
Info : Test测试123   [...\common\samples\debuglog.cpp(20) test]
*/
