/*
@brief Character encoding conversion
@author guobao.v@gmail.com
@note   Unicode(宽字符集)和ANSI(多字节字符集)是编码标准
        ANSI以单字节(8bit)存放英文字符，以双字节(16bit)存放中文等字符
        Unicode下，英文和中文的字符都以双字节存放(16bit)

        以下3个是Unicode标准的实现方式:
        UTF-16用2个字节来编码所有的字符(标准Unicode)
        UTF-32则选择用4个字节来编码
        UTF-8为了网络传输节省资源,采用变长编码(1-6字节编码)

        std::string  可以是ANSI编码(char 8bit,中文16bit(gb2312))
                     也可以是UTF-8编码(窄多字节编码，中文24bit，需转为ANSI或UNICODE后才能正常显示)
        std::wstring 可以是标准Unicode编码(wchar_t 16bit)
                     也可以是UTF-8编码(宽多字节编码)
        wchar_t		 在windows的MSVC环境下为UTF-16编码
                     在linux的GCC环境下为UTF-32编码
        char16_t	 UTF-16编码
        char32_t	 UTF-32编码
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_CODECVT_HPP_
#define _COMMON_CODECVT_HPP_

#include <common/precomm.hpp>
#include <codecvt>
#include <tchar.h>

//TODO 使用函数模板统一接口进行直接的互转
///std::string(utf8) std::string(ansi) std::u16string(utf16) std::u32string(utf32) std::wstring(unicode)

namespace common {

    namespace codecvt {

        /**
        *@brief 本地编码转换器
        */
        class local_codecvt : public std::codecvt_byname<wchar_t, char, std::mbstate_t> {
        public:
#ifdef _MSC_VER
            local_codecvt() : codecvt_byname("zh-CN") {}//设置本地语言环境
#else
            local_codecvt() : codecvt_byname("zh_CN.GB18030") {}
#endif
        };

        ///std::string(utf8) std::u16string(utf16) std::u32string(utf32)
        /**
        *@brief std::u16string -> std::string(utf8)
        *@return 若失败返回空字符串
        *@note 若包含中文需要将UTF-8转回多字符ANSI或宽字符Unicode才可正常显示中文.
        */
        static std::string utf16_to_utf8(const std::u16string& utf16_string) noexcept
        {
            std::string result = u8"";
#if _MSC_VER >= 1900
            std::wstring_convert<std::codecvt_utf8_utf16<int16_t>, int16_t> cvt;
            auto p = reinterpret_cast<const int16_t*>(utf16_string.data());
            try {
                result = cvt.to_bytes(p, p + utf16_string.size());
            }
#else
            std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> cvt;
            try {
                result = cvt.to_bytes(utf16_string);
            }
#endif
            catch (const std::range_error&) {
                return u8"";
            }
            return result;
        }

        /**
        *@brief std::string(utf8) -> std::u16string
        *@return 若失败返回空字符串
        */
        static std::u16string utf8_to_utf16(const std::string & utf8_string) noexcept
        {
            std::u16string result = u"";
#if _MSC_VER >= 1900
            std::wstring_convert<std::codecvt_utf8_utf16<int16_t>, int16_t> cvt;
            auto p = reinterpret_cast<const char*>(utf8_string.data());
            try {
                auto str = cvt.from_bytes(p, p + utf8_string.size());
                result.assign(str.begin(), str.end());
            }
#else
            std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> cvt;
            try
            {
                result = cvt.from_bytes(utf8_string);
            }
#endif
            catch (const std::range_error&) {
                return u"";
            }
            return result;
        }

        /**
        *@brief std::u32string -> std::string(utf8)
        *@return 若失败返回空字符串
        *@note 若包含中文需要将UTF-8转回多字符ANSI或宽字符Unicode才可正常显示中文.
        */
        static std::string utf32_to_utf8(const std::u32string & utf32_string) noexcept
        {
            std::string result = u8"";
#if _MSC_VER >= 1900
            std::wstring_convert<std::codecvt_utf8_utf16<int32_t>, int32_t> cvt;
            auto p = reinterpret_cast<const int32_t*>(utf32_string.data());
            try {
                result = cvt.to_bytes(p, p + utf32_string.size());
            }
#else
            std::wstring_convert<std::codecvt_utf8_utf16<char32_t>, char32_t> cvt;
            try {
                result = cvt.to_bytes(utf32_string);
            }
#endif
            catch (const std::range_error&) {
                return u8"";
            }
            return result;
        }

        /**
        *@brief std::string(utf8) -> std::u32string
        *@return 若失败返回空字符串
        */
        static std::u32string utf8_to_utf32(const std::string & utf8_string) noexcept
        {
            std::u32string result = U"";
#if _MSC_VER >= 1900
            std::wstring_convert<std::codecvt_utf8_utf16<int32_t>, int32_t> cvt;
            auto p = reinterpret_cast<const char*>(utf8_string.data());
            try {
                auto str = cvt.from_bytes(p, p + utf8_string.size());
                result.assign(str.begin(), str.end());
            }
#else
            std::wstring_convert<std::codecvt_utf8_utf16<char32_t>, char32_t> cvt;
            try {
                result = cvt.from_bytes(utf8_string);
            }
#endif
            catch (const std::range_error&) {
                return U"";
            }
            return result;
        }

        /**
        *@brief std::u32string -> std::u16string
        *@return 若失败返回空字符串
        */
        static std::u16string utf32_to_utf16(const std::u32string & utf32_string) noexcept
        {
            return utf8_to_utf16(utf32_to_utf8(utf32_string));
        }

        /**
        *@brief std::u16string -> std::u32string
        *@return 若失败返回空字符串
        */
        static std::u32string utf16_to_utf32(const std::u16string & utf16_string) noexcept
        {
            return utf8_to_utf32(utf16_to_utf8(utf16_string));
        }

        ///std::string(utf8) std::string(ansi) std::wstring(unicode)

        /**
        *@brief std::wstring(unicode) -> std::string(utf8)
        *@return 若失败返回空字符串,
        *@note 若包含中文需要将UTF-8转回多字符ANSI或宽字符Unicode才可正常显示中文.
        */
        static std::string unicode_to_utf8(const std::wstring & wstring) noexcept
        {
            std::string result = u8"";
            try {
                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> cvt;
                result = cvt.to_bytes(wstring);
            }
            catch (const std::range_error&) {
                return u8"";
            }
            return result;
        }

        /**
        *@brief std::string(utf8) -> std::wstring(unicode)
        *@return 若失败返回空字符串
        */
        static std::wstring utf8_to_unicode(const std::string & utf8_string) noexcept
        {
            std::wstring result = L"";
            try {
                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>, wchar_t> cvt;
                result = cvt.from_bytes(utf8_string);
            }
            catch (const std::range_error&) {
                return L"";
            }
            return result;
        }

        /**
        *@brief std::wstring(unicode) -> std::string(ansi)
        *@return 若失败返回空字符串
        */
        static std::string unicode_to_ansi(const std::wstring & wstring) noexcept
        {
            std::string result = "";
            std::wstring_convert<local_codecvt> cvt;
            try
            {
                result = cvt.to_bytes(wstring);
            }
            catch (const std::range_error&) {
                return u8"";
            }
            return result;
        }

        /**
        *@brief std::string(ansi) -> std::wstring(unicode)
        *@return 若失败返回空字符串
        */
        static std::wstring ansi_to_unicode(const std::string & ansi_string) noexcept
        {
            std::wstring result = L"";
            std::wstring_convert<local_codecvt> cvt;
            try
            {
                result = cvt.from_bytes(ansi_string);
            }
            catch (const std::range_error&) {
                return L"";
            }
            return result;
        }

        /**
        *@brief std::string(utf8) -> std::string(ansi)
        *@return 若失败返回空字符串
        */
        static std::string utf8_to_ansi(const std::string & utf8_string) noexcept
        {
            return unicode_to_ansi(utf8_to_unicode(utf8_string));
        }

        /**
        *@brief std::string(ansi) -> std::string(utf8)
        *@return 若失败返回空字符串
        *@note 若包含中文需要将UTF-8转回多字符ANSI或宽字符Unicode才可正常显示中文.
        */
        static std::string ansi_to_utf8(const std::string & ansi_string) noexcept
        {
            return unicode_to_utf8(ansi_to_unicode(ansi_string));
        }

        //std::string utf8_to_ansi(const std::wstring& utf8_wstring) noexcept {}
        //std::wstring ansi_to_utf8(const std::string& ansi_string) noexcept {}
        //std::wstring utf8_to_unicode(const std::wstring& utf8_wstring) noexcept {}
        //std::wstring unicode_to_utf8(const std::wstring& unicode_wstring) noexcept {}

    }// namespace codecvt

}// namespace common

#endif // _COMMON_CODECVT_HPP_

