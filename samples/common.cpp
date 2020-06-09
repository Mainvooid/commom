/*
@brief usage sample for namespace common.
@author guobao.v@gmail.com
*/
#pragma warning(disable:4091)
#include <common.hpp>

size_t __stdcall addA(const char* str,size_t a, size_t b) { return a + b; };
size_t addW(const wchar_t* str, size_t a, size_t b) { return a + b; };

template<typename T>
auto addAW(const T* str) {
    return tvalue<T>(getFunction(addA), getFunction(addW))(str,2,3);
}

int main() {
    //目录路径填充
    std::wstring wstr;
    wstr = common::fillDir(L"D:\\a\\b\\c\\", L"/");//"D:/a/b/c/"
    wstr = common::fillDir(L"D:\\a\\b\\c", L"/");  //"D:/a/b/c/"
    wstr = common::fillDir(L"D:/a/b/c/", L"\\");   //"D:\a\b\c\"
    wstr = common::fillDir(L"D:/a/b/c", L"\\");    //"D:\a\b\c\"
    wstr = common::fillDir(L"a\\b\\c", L"/");      //"a/b/c/"

    //函数计时
    std::function<std::wstring(const wchar_t*, const wchar_t*)> fn = fillDir<wchar_t>;
    cout << getFnDuration<std::chrono::microseconds>(fn, L"D:\\a\\b\\c\\", L"/") << endl; //6us

    DWORD delay = 100;
    std::function<void()> fn2 = [&]()->void {::Sleep(delay); };
    cout << getFnDuration<std::chrono::seconds>(fn2) << endl;      //0s
    cout << getFnDuration<std::chrono::milliseconds>(fn2) << endl; //100ms
    cout << getFnDuration<std::chrono::microseconds>(fn2) << endl; //101,020us
    cout << getFnDuration<std::chrono::nanoseconds>(fn2) << endl;  //100,203,300ns

    //根据模板参数进行值推断
    const char* str = tvalue<char>("ansi", L"wide");  //ansi
    wstr = tvalue<wchar_t>("ansi", L"wide");          //wide
    std::string string = tvalue<std::string>("string", L"wstring");    //string
    std::wstring wstring = tvalue<std::wstring>("string", L"wstring"); //wstring

    //根据模板参数调用函数并获取结果值
    std::function<size_t(size_t, size_t)> add = [&](size_t x, size_t y) {return x + y; };
    std::function<int(int, int)> mul = [&](int x, int y) {return x * y; };
    int fn_result = tvalue<wchar_t>(add, mul)(2, 3); //6
    fn_result = tvalue<std::wstring>([&](size_t x, size_t y) {return x + y; }, [&](int x, int y) {return x * y; })(2, 3); //5
    auto sum = addAW("test"); //5

    //字符数组长度
    size_t n = cslen(str);  //4
    n = cslen(wstr.data()); //4

    //convert to string
    double d = 1.23;
    std::string cstr = convert_to_string<char>(d);//"1.23"
    float f = convert_from_string<float>(cstr);   //1.23000002
}
