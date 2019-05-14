/*
@brief usage sample for common.hpp.
@author guobao.v@gmail.com
*/
#include <common/common.hpp>
#include <windows.h>
using namespace common;
using namespace std;

int main() {
    std::wstring wstr;
    wstr = fillDir(L"D:\\a\\b\\c\\", L"/");//"D:/a/b/c/"
    wstr = fillDir(L"D:\\a\\b\\c", L"/");  //"D:/a/b/c/"
    wstr = fillDir(L"D:/a/b/c/", L"\\");   //"D:\a\b\c\"
    wstr = fillDir(L"D:/a/b/c", L"\\");    //"D:\a\b\c\"
    wstr = fillDir(L"a\\b\\c", L"/");      //"a/b/c/"

    std::function<std::wstring(const wchar_t*, const wchar_t*)> fn = fillDir<wchar_t>;
    cout << getFnDuration<std::chrono::microseconds>(fn, L"D:\\a\\b\\c\\", L"/") << endl; //5us

    size_t delay = 100;
    std::function<void()> fn2 = [&]()->void {::Sleep(delay); };
    cout << getFnDuration<std::chrono::seconds>(fn2) << endl;      //0s
    cout << getFnDuration<std::chrono::milliseconds>(fn2) << endl; //100ms
    cout << getFnDuration<std::chrono::microseconds>(fn2) << endl; //101,020us
    cout << getFnDuration<std::chrono::nanoseconds>(fn2) << endl;  //100,203,300ns

}