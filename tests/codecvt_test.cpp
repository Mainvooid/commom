/*
@brief unit test for codecvt.hpp
@author guobao.v@gmail.com
*/
#include "gtest/gtest.h"
#include <common.hpp>
#include <common/codecvt.hpp>
using namespace common::codecvt;

//\u0054\u0065\u0073\u0074\u6d4b\u8bd5\u0031\u0032\u0033\u002e\u000A
std::string a_s = "Test测试123.\n";
std::string u8_s = u8"Test测试123.\n";
std::wstring w_s = L"Test测试123.\n";
std::u16string u16_s = u"Test测试123.\n";
std::u32string u32_s = U"Test测试123.\n";

//std::string(utf8) std::u16string(utf16) std::u32string(utf32)
TEST(codecvt, utf16_to_utf8) {
    EXPECT_EQ(utf16_to_utf8(u16_s), u8_s);
}
TEST(codecvt, utf8_to_utf16) {
    EXPECT_EQ(utf8_to_utf16(u8_s), u16_s);
}
TEST(codecvt, utf32_to_utf8) {
    EXPECT_EQ(utf32_to_utf8(u32_s), u8_s);
}
TEST(codecvt, utf8_to_utf32) {
    EXPECT_EQ(utf8_to_utf32(u8_s), u32_s);
}
TEST(codecvt, utf32_to_utf16) {
    EXPECT_EQ(utf32_to_utf16(u32_s), u16_s);
}
TEST(codecvt, utf16_to_utf32) {
    EXPECT_EQ(utf16_to_utf32(u16_s), u32_s);
}

//std::string(utf8) std::string(ansi) std::wstring(unicode)
TEST(codecvt, utf8_to_ansi) {
    EXPECT_EQ(utf8_to_ansi(u8_s), a_s);
}
TEST(codecvt, ansi_to_utf8) {
    EXPECT_EQ(ansi_to_utf8(a_s), u8_s);
}
TEST(codecvt, unicode_to_utf8) {
    EXPECT_EQ(unicode_to_utf8(w_s), u8_s);
}
TEST(codecvt, utf8_to_unicode) {
    EXPECT_EQ(utf8_to_unicode(u8_s), w_s);
}
TEST(codecvt, unicode_to_ansi) {
    EXPECT_EQ(unicode_to_ansi(w_s), a_s);
}
TEST(codecvt, ansi_to_unicode) {
    EXPECT_EQ(ansi_to_unicode(a_s), w_s);
}

//to_unicode() to_ansi() to_utf8()
TEST(codecvt, ato_unicode) {
    EXPECT_EQ(to_unicode(a_s), w_s);
}
TEST(codecvt, wto_unicode) {
    EXPECT_EQ(to_unicode(w_s), w_s);
}
TEST(codecvt, ato_ansi) {
    EXPECT_EQ(to_ansi(a_s), a_s);
}
TEST(codecvt, wto_ansi) {
    EXPECT_EQ(to_ansi(w_s), a_s);
}
TEST(codecvt, ato_utf8) {
    EXPECT_EQ(to_utf8(a_s), u8_s);
}
TEST(codecvt, wto_utf8) {
    EXPECT_EQ(to_utf8(w_s), u8_s);
}
TEST(codecvt, u16to_utf8) {
    EXPECT_EQ(to_utf8(u16_s), u8_s);
}
TEST(codecvt, u32to_utf8) {
    EXPECT_EQ(to_utf8(u32_s), u8_s);
}