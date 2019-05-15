/*
@brief usage sample for namespace common.
@author guobao.v@gmail.com
*/
#include <common/precomm.hpp>
#include <common.hpp>
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
    cout << getFnDuration<std::chrono::microseconds>(fn, L"D:\\a\\b\\c\\", L"/") << endl; //6us

    size_t delay = 100;
    std::function<void()> fn2 = [&]()->void {::Sleep(delay); };
    cout << getFnDuration<std::chrono::seconds>(fn2) << endl;      //0s
    cout << getFnDuration<std::chrono::milliseconds>(fn2) << endl; //100ms
    cout << getFnDuration<std::chrono::microseconds>(fn2) << endl; //101,020us
    cout << getFnDuration<std::chrono::nanoseconds>(fn2) << endl;  //100,203,300ns

    ttype<char, char, wchar_t>::type;    //char
    ttype<wchar_t, char, wchar_t>::type; //wchar_t

    const char* str = tvalue<char>("ansi", L"wide");  //ansi
    wstr = tvalue<wchar_t>("ansi", L"wide");          //wide
    size_t n = cslen(str);  //4
    n = cslen(wstr.data()); //4

}