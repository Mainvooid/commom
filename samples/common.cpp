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
    //目录路径填充
    std::wstring wstr;
    wstr = fillDir(L"D:\\a\\b\\c\\", L"/");//"D:/a/b/c/"
    wstr = fillDir(L"D:\\a\\b\\c", L"/");  //"D:/a/b/c/"
    wstr = fillDir(L"D:/a/b/c/", L"\\");   //"D:\a\b\c\"
    wstr = fillDir(L"D:/a/b/c", L"\\");    //"D:\a\b\c\"
    wstr = fillDir(L"a\\b\\c", L"/");      //"a/b/c/"

    //函数计时
    std::function<std::wstring(const wchar_t*, const wchar_t*)> fn = fillDir<wchar_t>;
    cout << getFnDuration<std::chrono::microseconds>(fn, L"D:\\a\\b\\c\\", L"/") << endl; //6us

    size_t delay = 100;
    std::function<void()> fn2 = [&]()->void {::Sleep(delay); };
    cout << getFnDuration<std::chrono::seconds>(fn2) << endl;      //0s
    cout << getFnDuration<std::chrono::milliseconds>(fn2) << endl; //100ms
    cout << getFnDuration<std::chrono::microseconds>(fn2) << endl; //101,020us
    cout << getFnDuration<std::chrono::nanoseconds>(fn2) << endl;  //100,203,300ns

    //条件类型判断
    ttype<char, char, wchar_t>::type;    //char
    ttype<wchar_t, char, wchar_t>::type; //wchar_t
    ttype<std::string, std::string, std::wstring>::type;  //std::string
    ttype<std::wstring, std::string, std::wstring>::type; //std::wstring
    ttype<std::string, std::function<size_t(size_t, size_t)>, std::function<int(int, int)>>::type; //std::function<size_t(size_t, size_t)>

    //根据模板参数进行值推断
    const char* str = tvalue<char>("ansi", L"wide");  //ansi
    wstr = tvalue<wchar_t>("ansi", L"wide");          //wide
    std::string string = tvalue<std::string>("string", L"wstring");    //string
    std::wstring wstring = tvalue<std::wstring>("string", L"wstring"); //wstring

    //根据模板参数调用函数并获取结果值
    std::function<size_t(size_t, size_t)> add = [&](size_t x, size_t y) {return x + y; };
    std::function<int(int, int)> mul = [&](int x, int y) {return x * y; };
    int fn_result = tvalue<wchar_t>(add, mul)(2, 3); //6
    fn_result = tvalue<std::string>([&](size_t x, size_t y) {return x + y; }, [&](int x, int y) {return x * y; })(2, 3); //5

    //字符数组长度
    size_t n = cslen(str);  //4
    n = cslen(wstr.data()); //4
}