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

    /*调试日志级别
    Trace    = 0, // 更细粒度的消息记录
    Debug    = 1, // 细粒度调试信息事件
    Info     = 2, // 粗粒度记录应用程序的正常运行过程
    Warn     = 3, // 可能导致潜在错误
    Error    = 4, // 不影响系统继续运行的错误事件
    Fatal    = 5, // 会导致应用程序退出的致命事件
    Off      = 6  // 关闭日志记录
    */
    enum level_e {
        Trace = 0,
        Debug = 1,
        Info = 2,
        Warn = 3,
        Error = 4,
        Fatal = 5,
        Off = 6
    };

    namespace debuglog {

        /**
        *@brief OutputDebugString扩展版
        *@param format 格式 e.g. L"Error [%s] : %s\n"
        *@param ... 相应参数包
        */
        template<typename T>
        static void OutputDebugStringEx(const T* format, ...) noexcept
        {
            T buf[BUFSIZ];
            va_list args;//可变长参数列表
            va_start(args, format);//获取列表第一个参数
            _vstprintf_s(buf, format, args);//按格式执行拼接
            va_end(args);//清空列表
            ::OutputDebugString(buf);
        }

        /*
       *@brief 调试日志类
       */
        template<typename T =
#if defined(_UNICODE) or defined(UNICODE)
            std::wstring
#else
            std::string
#endif
        >
            class logger
        {
        public:
            logger(T logger_name = _T(""), level_e log_level = level_e::Error)
                :m_name(std::move(logger_name)), m_level(log_level) {}
            virtual ~logger() {}
            logger(const logger & logger) :m_name(logger.m_name), m_level(logger.m_level) {};
            logger &operator=(const logger & logger) { m_name = logger.m_name; m_level = logger.m_level; return *this; }
            logger(const logger&&) = delete;
            logger& operator=(const logger&&) = delete;
        public:
            void setLevel(level_e log_level) { m_level = log_level; }
            level_e getLevel() const { return m_level; }
            void setName(T logger_name) { m_name = std::move(logger_name); }
            const T& getName() const { return m_name; }

            /*
            *@brief 打印Debug信息
            */
            void Log(T msg, const level_e nLevel) const
            {
                if (nLevel < m_level || nLevel == level_e::Off) { return; }
                T _name = m_name;
                if (m_name == _T("")) { _name = _T("G"); }

                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx(_T("Trace [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Debug:
                    OutputDebugStringEx(_T("Debug [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Info:
                    OutputDebugStringEx(_T("Info  [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Warn:
                    OutputDebugStringEx(_T("Warn  [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Error:
                    OutputDebugStringEx(_T("Error [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Fatal:
                    OutputDebugStringEx(_T("Fatal [%s] : %s\n"), _name.data(), msg.data());
                    break;
                default:
                    break;
                }
            }

            /*
            *@brief 打印Debug信息
            *@param _func 宏 __FUNCTION__ 函数名
            *@param _file 宏 __FILE__ 文件名
            *@param _line 宏 __LINE__ 行数
            */
            void Log(T msg, const level_e nLevel, const char* _func, const char * _file, const int _line)const
            {
                if (nLevel < m_level || nLevel == level_e::Off) { return; }
                T func, file;
#if defined(_UNICODE) or defined(UNICODE)
                if (std::is_same<T, std::wstring>::value) {
                    func = codecvt::ansi_to_unicode(_func);
                    file = codecvt::ansi_to_unicode(_file);
                }
#else
                if (std::is_same<T, std::string>::value) {
                    func = _func;
                    file = _file;
                }
#endif
                size_t n = file.find_last_of('\\');
                n = file.find_last_of('\\', n - 1);
                n = file.find_last_of('\\', n - 1);
                if (n == static_cast<size_t>(-1)) { n = 0; }
                T file_fmt = T(file.begin() + n, file.end());

                T _name = m_name;
                if (m_name == _T("")) { _name = _T(" "); }

                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx(_T("Trace [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Debug:
                    OutputDebugStringEx(_T("Debug [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Info:
                    OutputDebugStringEx(_T("Info  [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Warn:
                    OutputDebugStringEx(_T("Warn  [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Error:
                    OutputDebugStringEx(_T("Error [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Fatal:
                    OutputDebugStringEx(_T("Fatal [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                default:
                    break;
                }
            }

            template<typename T>
            void Trace(T msg) const { Log(std::move(msg), level_e::Trace); }
            template<typename T>
            void Debug(T msg) const { Log(std::move(msg), level_e::Debug); }
            template<typename T>
            void Info(T msg) const { Log(std::move(msg), level_e::Info); }
            template<typename T>
            void Warn(T msg) const { Log(std::move(msg), level_e::Warn); }
            template<typename T>
            void Error(T msg) const { Log(std::move(msg), level_e::Error); }
            template<typename T>
            void Fatal(T msg) const { Log(std::move(msg), level_e::Fatal); }

            template<typename T>
            void Trace(T msg, const char* _func, const char * _file, const int _line) const { Log(std::move(msg), level_e::Trace, _func, _file, _line); }
            template<typename T>
            void Debug(T msg, const char* _func, const char * _file, const int _line) const { Log(std::move(msg), level_e::Debug, _func, _file, _line); }
            template<typename T>
            void Info(T msg, const char* _func, const char * _file, const int _line) const { Log(std::move(msg), level_e::Info, _func, _file, _line); }
            template<typename T>
            void Warn(T msg, const char* _func, const char * _file, const int _line) const { Log(std::move(msg), level_e::Warn, _func, _file, _line); }
            template<typename T>
            void Error(T msg, const char* _func, const char * _file, const int _line) const { Log(std::move(msg), level_e::Error, _func, _file, _line); }
            template<typename T>
            void Fatal(T msg, const char* _func, const char * _file, const int _line) const { Log(std::move(msg), level_e::Fatal, _func, _file, _line); }

        private:
            T m_name;
            level_e m_level;
        };

    } // namespace debuglog

    /// common 全局logger
    static std::unique_ptr<debuglog::logger<>> g_logger(new debuglog::logger<>(_T("G"), level_e::Trace));

    template<typename T>
    void LOGT(T msg) { g_logger->Trace(std::move(msg)); };
    template<typename T>
    void LOGD(T msg) { g_logger->Debug(std::move(msg)); };
    template<typename T>
    void LOGI(T msg) { g_logger->Info(std::move(msg)); };
    template<typename T>
    void LOGW(T msg) { g_logger->Warn(std::move(msg)); };
    template<typename T>
    void LOGE(T msg) { g_logger->Error(std::move(msg)); };
    template<typename T>
    void LOGF(T msg) { g_logger->Fatal(std::move(msg)); };

    template<typename T>
    void LOGT(T msg, const char* _func, const char * _file, const int _line) { g_logger->Trace(std::move(msg), _func, _file, _line); };
    template<typename T>
    void LOGD(T msg, const char* _func, const char * _file, const int _line) { g_logger->Debug(std::move(msg), _func, _file, _line); };
    template<typename T>
    void LOGI(T msg, const char* _func, const char * _file, const int _line) { g_logger->Info(std::move(msg), _func, _file, _line); };
    template<typename T>
    void LOGW(T msg, const char* _func, const char * _file, const int _line) { g_logger->Warn(std::move(msg), _func, _file, _line); };
    template<typename T>
    void LOGE(T msg, const char* _func, const char * _file, const int _line) { g_logger->Error(std::move(msg), _func, _file, _line); };
    template<typename T>
    void LOGF(T msg, const char* _func, const char * _file, const int _line) { g_logger->Fatal(std::move(msg), _func, _file, _line); };

} // namespace common

#endif // _COMMON_DEBUGLOG_HPP_

