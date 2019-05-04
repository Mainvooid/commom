/*
@brief usage sample for codecvt.h.
@author guobao.v@gmail.com
*/
#include <common/codecvt.h>
using namespace common::codecvt;

int main() {
    //\u0054\u0065\u0073\u0074\u6d4b\u8bd5\u0031\u0032\u0033\u002e\u005c\u006e
    const std::string a_s = "Test测试123.\n";
    const std::string u8_s = u8"Test测试123.\n";
    const std::wstring w_s = L"Test测试123.\n";
    const std::u16string u16_s = u"Test测试123.\n";
    const std::u32string u32_s = U"Test测试123.\n";
    std::string dst_a_s;
    std::string dst_u8_s;
    std::wstring dst_w_s;
    std::u16string dst_u16_s;
    std::u32string dst_u32_s;

    ///std::string(utf8) std::u16string(utf16) std::u32string(utf32)
    dst_u8_s = utf16_to_utf8(u16_s);  //Test娴嬭瘯123.\n
    dst_u16_s = utf8_to_utf16(u8_s);  //Test测试123.\n
    dst_u8_s = utf32_to_utf8(u32_s);  //Test娴嬭瘯123.\n
    dst_u32_s = utf8_to_utf32(u8_s);  //Test测试123.\n
    dst_u16_s = utf32_to_utf16(u32_s);//Test测试123.\n
    dst_u32_s = utf16_to_utf32(u16_s);//Test测试123.\n
    ///std::string(utf8) std::string(ansi) std::wstring(unicode)
    dst_a_s = utf8_to_ansi(u8_s);   //Test测试123.\n
    dst_u8_s = ansi_to_utf8(a_s);   //Test娴嬭瘯123.\n
    dst_u8_s = unicode_to_utf8(w_s);//Test娴嬭瘯123.\n
    dst_w_s = utf8_to_unicode(u8_s);//Test测试123.\n
    dst_a_s = unicode_to_ansi(w_s); //Test测试123.\n
    dst_w_s = ansi_to_unicode(a_s); //Test测试123.\n
}