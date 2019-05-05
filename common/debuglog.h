/*
@brief a simple debug logger
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _DEBUGLOG_H_
#define _DEBUGLOG_H_

#ifndef _WIN32
#error Unsupported platform
#else

#include <common/codecvt.h>
#include <windows.h>

namespace common {
    namespace debuglog {
        const std::string _TAG = "debuglog";

        /*调试日志级别
        Trace    = 0,
        Debug    = 1,
        Info     = 2,
        Warn     = 3,
        Error    = 4,
        Critical = 5
        */
        enum level_e {
            Trace = 0,
            Debug = 1,
            Info = 2,
            Warn = 3,
            Error = 4,
            Critical = 5
        };

        /*
       *@brief 调试日志类
       */
        class logger
        {
        public:
            logger() :m_name(_T("")), m_level(level_e::Warn) {}
            ~logger() noexcept {}
            logger(std::wstring logger_name) :m_name(std::move(logger_name)), m_level(level_e::Warn) {}
            logger(level_e log_level) :m_name(_T("")), m_level(log_level) { }
            logger(std::wstring logger_name, level_e log_level) :m_name(std::move(logger_name)), m_level(log_level) { }
            logger(const logger &) = delete;           //拷贝构造
            logger &operator=(const logger &) = delete;//拷贝赋值
            logger(const logger&&) = delete;           //移动构造
            logger& operator=(const logger&&) = delete;//移动赋值
        public:
            void setLevel(level_e log_level) noexcept { m_level = log_level; }

            /*
            *@brief 打印Debug信息
            *@param wmsgbuf 提示信息
            *@param nLevel log等级
            */
            void printLog(const wchar_t * wmsgbuf, const level_e nLevel) const noexcept
            {
                if (nLevel < m_level) { return; }
                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx(L"Trace [%s] : %s\n",m_name.data(), wmsgbuf);
                    break;
                case level_e::Debug:
                    OutputDebugStringEx(L"Debug [%s] : %s\n", m_name.data(), wmsgbuf);
                    break;
                case level_e::Info:
                    OutputDebugStringEx(L"Info [%s] : %s\n", m_name.data(), wmsgbuf);
                    break;
                case level_e::Warn:
                    OutputDebugStringEx(L"Warn [%s] : %s\n", m_name.data(), wmsgbuf);
                    break;
                case level_e::Error:
                    OutputDebugStringEx(L"Error [%s] : %s\n", m_name.data(), wmsgbuf);
                    break;
                case level_e::Critical:
                    OutputDebugStringEx(L"Critical [%s] : %s\n", m_name.data(), wmsgbuf);
                    break;
                default:
                    break;
                }
            }

            /*
            *@brief 打印Debug信息
            *@param msgbuf 提示信息
            *@param nLevel log等级
            */
            void printLog(const std::string msgbuf, const level_e nLevel) const noexcept
            {
                if (nLevel < m_level) { return; }
                std::wstring wstring = codecvt::ansi_to_unicode(msgbuf);
                printLog(wstring.data(), nLevel);
            }

            /*
            *@brief 打印Debug信息
            *@param wmsgbuf 提示信息
            *@param nLevel log等级
            *@param _func 宏 __FUNCTION__ 函数名
            *@param _file 宏 __FILE__ 文件名
            *@param _line 宏 __LINE__ 行数
            */
            void printLog(const wchar_t * wmsgbuf, const level_e nLevel,
                const char* _func, const char * _file, const int _line)const noexcept
            {
                if (nLevel < m_level) { return; }
                std::wstring wfunc = codecvt::ansi_to_unicode(_func);
                std::wstring wfile = codecvt::ansi_to_unicode(_file);

                size_t n = wfile.find_last_of('\\');
                n = wfile.find_last_of('\\', n - 1);
                n = wfile.find_last_of('\\', n - 1);
                std::wstring wfile_fmt = std::wstring(wfile.begin() + n, wfile.end());

                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx(_T("Trace [%s] : %s   [...%s(%d) %s]\n"),
                        m_name.data(), wmsgbuf, wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Debug:
                    OutputDebugStringEx(_T("Debug [%s] : %s   [...%s(%d) %s]\n"),
                        m_name.data(), wmsgbuf, wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Info:
                    OutputDebugStringEx(_T("Info [%s] : %s   [...%s(%d) %s]\n"),
                        m_name.data(), wmsgbuf, wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Warn:
                    OutputDebugStringEx(_T("Warn [%s] : %s   [...%s(%d) %s]\n"),
                        m_name.data(), wmsgbuf, wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Error:
                    OutputDebugStringEx(_T("Error [%s] : %s   [...%s(%d) %s]\n"),
                        m_name.data(), wmsgbuf, wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Critical:
                    OutputDebugStringEx(_T("Critical [%s] : %s   [...%s(%d) %s]\n"),
                        m_name.data(), wmsgbuf, wfile_fmt.data(), _line, wfunc.data());
                    break;
                default:
                    break;
                }
            }

            /*
            *@brief 打印Debug信息
            *@param msgbuf 提示信息
            *@param nLevel log等级
            *@param _func 宏 __FUNCTION__ 函数名
            *@param _file 宏 __FILE__ 文件名
            *@param _line 宏 __LINE__ 行数
            */
            void printLog(const std::string msgbuf, const level_e nLevel,
                const char* _func, const char * _file, const int _line)const noexcept
            {
                if (nLevel < m_level) { return; }
                std::wstring wmsgbuf = codecvt::ansi_to_unicode(msgbuf);
                printLog(wmsgbuf.data(), nLevel, _func, _file, _line);
            }

        private:
            //TODO 当前是C风格的,有待改进
            /**
            *@brief OutputDebugStringW扩展版
            *@param format 格式 e.g."Error : %s [%s(%d)]\n"
            *@param ... 相应参数包
            */
            void OutputDebugStringEx(const wchar_t * format, ...)const noexcept
            {
                wchar_t buf[BUFSIZ];
                va_list args;//可变长参数列表
                va_start(args, format);//获取列表第一个参数
                int len = _vstprintf_s(buf, format, args);//按格式执行拼接
                va_end(args);//清空列表
                ::OutputDebugStringW(buf);
            }
        private:
            std::wstring m_name;
            level_e m_level;
        };

    } // namespace debuglog

} // namespace common

#endif // _WIN32

#endif // _DEBUGLOG_H_

