/*
@brief a simple debug logger
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#if !defined(_COMMON_DEBUGLOG_HPP_) && defined(_WIN32)
#define _COMMON_DEBUGLOG_HPP_

#include <common/codecvt.hpp>
#include <windows.h>

namespace common {

    namespace debuglog {

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

        //TODO 当前是C风格的,有待改进
        /**
        *@brief OutputDebugStringW扩展版
        *@param format 格式 e.g."Error : %s [%s(%d)]\n"
        *@param ... 相应参数包
        */
        static void OutputDebugStringEx(const wchar_t * format, ...)
        {
            wchar_t buf[BUFSIZ];
            va_list args;//可变长参数列表
            va_start(args, format);//获取列表第一个参数
            int len = _vstprintf_s(buf, format, args);//按格式执行拼接 //TODO 多余的局部变量
            va_end(args);//清空列表
            ::OutputDebugStringW(buf);
        }

        /*
       *@brief 调试日志类
       */
        class logger
        {
        public:
            logger(std::wstring logger_name = _T(""), level_e log_level = level_e::Error)
                :m_name(std::move(logger_name)), m_level(log_level) {}
            ~logger() noexcept {}
            logger(const logger & logger) :m_name(logger.m_name), m_level(logger.m_level) {};
            logger &operator=(const logger & logger) { m_name = logger.m_name; m_level = logger.m_level; return *this; }
            logger(const logger&&) = delete;
            logger& operator=(const logger&&) = delete;
        public:
            void setLevel(level_e log_level) noexcept { m_level = log_level; }
            level_e getLevel() noexcept { return m_level; }
            void setName(std::wstring logger_name) noexcept { m_name = logger_name; }
            std::wstring getName() noexcept { return m_name; }

            /*
            *@brief 打印Debug信息
            *@param wmsgbuf 提示信息
            *@param nLevel log等级
            */
            void printLog(const std::wstring& wmsg, const level_e nLevel) const noexcept
            {
                if (nLevel < m_level) { return; }
                std::wstring _name = m_name;
                if (m_name == _T(""))
                {
                    _name = _T("G");
                }
                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx(L"Trace [%s] : %s\n", _name.data(), wmsg.data());
                    break;
                case level_e::Debug:
                    OutputDebugStringEx(L"Debug [%s] : %s\n", _name.data(), wmsg.data());
                    break;
                case level_e::Info:
                    OutputDebugStringEx(L"Info [%s] : %s\n", _name.data(), wmsg.data());
                    break;
                case level_e::Warn:
                    OutputDebugStringEx(L"Warn [%s] : %s\n", _name.data(), wmsg.data());
                    break;
                case level_e::Error:
                    OutputDebugStringEx(L"Error [%s] : %s\n", _name.data(), wmsg.data());
                    break;
                case level_e::Critical:
                    OutputDebugStringEx(L"Critical [%s] : %s\n", _name.data(), wmsg.data());
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
            void printLog(const std::wstring& wmsg, const level_e nLevel,
                const char* _func, const char * _file, const int _line)const noexcept
            {
                if (nLevel < m_level) { return; }
                std::wstring wfunc = codecvt::ansi_to_unicode(_func);
                std::wstring wfile = codecvt::ansi_to_unicode(_file);

                size_t n = wfile.find_last_of('\\');
                n = wfile.find_last_of('\\', n - 1);
                n = wfile.find_last_of('\\', n - 1);
                std::wstring wfile_fmt = std::wstring(wfile.begin() + n, wfile.end());
                std::wstring _name = m_name;
                if (m_name == _T(""))
                {
                    _name = _T("G");
                }

                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx(_T("Trace [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), wmsg.data(), wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Debug:
                    OutputDebugStringEx(_T("Debug [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), wmsg.data(), wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Info:
                    OutputDebugStringEx(_T("Info [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), wmsg.data(), wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Warn:
                    OutputDebugStringEx(_T("Warn [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), wmsg.data(), wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Error:
                    OutputDebugStringEx(_T("Error [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), wmsg.data(), wfile_fmt.data(), _line, wfunc.data());
                    break;
                case level_e::Critical:
                    OutputDebugStringEx(_T("Critical [%s] : %s   [...%s(%d) %s]\n"),
                        m_name.data(), wmsg.data(), wfile_fmt.data(), _line, wfunc.data());
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
            std::wstring m_name;
            level_e m_level;
        };

    } // namespace debuglog

    /// common 全局logger
    static debuglog::logger g_logger(L"", debuglog::level_e::Trace);

    static void LOGT(std::wstring wmsg) {
        g_logger.printLog(wmsg, debuglog::level_e::Trace);
    };
    static void LOGD(std::wstring wmsg) {
        g_logger.printLog(wmsg, debuglog::level_e::Debug);
    };
    static void LOGI(std::wstring wmsg) {
        g_logger.printLog(wmsg, debuglog::level_e::Info);
    };
    static void LOGW(std::wstring wmsg) {
        g_logger.printLog(wmsg, debuglog::level_e::Warn);
    };
    static void LOGE(std::wstring wmsg) {
        g_logger.printLog(wmsg, debuglog::level_e::Error);
    };
    static void LOGC(std::wstring wmsg) {
        g_logger.printLog(wmsg, debuglog::level_e::Critical);
    };

    static void LOGT(std::wstring wmsg, const char* _func, const char * _file, const int _line) {
        g_logger.printLog(wmsg, debuglog::level_e::Trace, _func, _file, _line);
    };
    static void LOGD(std::wstring wmsg, const char* _func, const char * _file, const int _line) {
        g_logger.printLog(wmsg, debuglog::level_e::Debug, _func, _file, _line);
    };
    static void LOGI(std::wstring wmsg, const char* _func, const char * _file, const int _line) {
        g_logger.printLog(wmsg, debuglog::level_e::Info, _func, _file, _line);
    };
    static void LOGW(std::wstring wmsg, const char* _func, const char * _file, const int _line) {
        g_logger.printLog(wmsg, debuglog::level_e::Warn, _func, _file, _line);
    };
    static void LOGE(std::wstring wmsg, const char* _func, const char * _file, const int _line) {
        g_logger.printLog(wmsg, debuglog::level_e::Error, _func, _file, _line);
    };
    static void LOGC(std::wstring wmsg, const char* _func, const char * _file, const int _line) {
        g_logger.printLog(wmsg, debuglog::level_e::Critical, _func, _file, _line);
    };

} // namespace common

#endif // _COMMON_DEBUGLOG_HPP_

