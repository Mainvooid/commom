/*
@file a simple debug logger
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4067)
#endif

#if !defined(_COMMON_DEBUGLOG_HPP_) && defined(_WIN32)
#define _COMMON_DEBUGLOG_HPP_

#include <common/codecvt.hpp>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <mutex>
#include <functional>
#include <windows.h>

/**
  @addtogroup common
  @{
    @defgroup debuglog debuglog - windows debug logger
  @}
*/
namespace common {
    /// @addtogroup common
    /// @{

    /**
    @brief 调试日志级别
    */
    enum level_e {
        Trace = 0, /*< 更细粒度的消息记录 */
        Debug = 1, /*< 细粒度调试信息事件 */
        Info  = 2, /*< 粗粒度记录应用程序的正常运行过程 */
        Warn  = 3, /*< 可能导致潜在错误 */
        Error = 4, /*< 不影响系统继续运行的错误事件 */
        Fatal = 5, /*< 会导致应用程序退出的致命事件 */
        Off   = 6  /*< 关闭日志记录 */
    };

    namespace debuglog {
        /// @addtogroup debuglog
        /// @{

        /**
        *@brief OutputDebugString扩展版
        *@param format 格式 e.g. "Error [%s] : %s\n"
        *@param ... 相应参数包
        */
        class OutputDebugStringEx
        {
        public:
            void operator()(const char* format, ...)
            {
                char buf[BUFSIZ];
                va_list args;
                va_start(args, format);
                vsprintf_s(buf, BUFSIZ, format, args);
                va_end(args);
                OutputDebugStringA(buf);
            };
            void operator()(const wchar_t* format, ...)
            {
                wchar_t buf[BUFSIZ];
                va_list args;
                va_start(args, format);
                vswprintf_s(buf, BUFSIZ, format, args);
                va_end(args);
                OutputDebugStringW(buf);
            };
        };

        /**
        *@brief 调试日志类
        */
        class logger
        {
        public:
            logger(std::string logger_name = "", level_e log_level = level_e::Error)
                :m_name(std::move(logger_name)), m_level(log_level) {}
            logger(std::wstring logger_name = L"", level_e log_level = level_e::Error)
                :m_name(std::move(codecvt::unicode_to_ansi(logger_name))), m_level(log_level) {}

            virtual ~logger() {}
            logger(const logger& logger) :m_name(logger.m_name), m_level(logger.m_level) {};
            logger& operator=(const logger& logger) { m_name = logger.m_name; m_level = logger.m_level; return *this; }
            logger(const logger&&) = delete;
            logger& operator=(const logger&&) = delete;
        public:
            void setLevel(level_e log_level) { m_level = log_level; }
            level_e getLevel() const { return m_level; }
            void setName(std::string logger_name) { m_name = std::move(logger_name); }
            void setName(std::wstring logger_name) { m_name = std::move(codecvt::unicode_to_ansi(logger_name)); }
            const std::string& getName() const { return m_name; }
            std::wstring getWName() const { return codecvt::ansi_to_unicode(m_name); }
            /*
            *@brief 打印Debug信息
            */
            template<typename T>
            void Log(T msg, const level_e nLevel) const
            {
                if (nLevel < m_level || nLevel == level_e::Off) { return; }
                T _name = tvalue<T>(m_name, codecvt::ansi_to_unicode(m_name));
                if (m_name == "") { _name = tvalue<T>("G", L"G"); }

                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx()(tvalue<T>("Trace [%s] : %s\n", L"Trace [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Debug:
                    OutputDebugStringEx()(tvalue<T>("Debug [%s] : %s\n", L"Debug [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Info:
                    OutputDebugStringEx()(tvalue<T>("Info  [%s] : %s\n", L"Info  [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Warn:
                    OutputDebugStringEx()(tvalue<T>("Warn  [%s] : %s\n", L"Warn  [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Error:
                    OutputDebugStringEx()(tvalue<T>("Error [%s] : %s\n", L"Error [%s] : %s\n"), _name.data(), msg.data());
                    break;
                case level_e::Fatal:
                    OutputDebugStringEx()(tvalue<T>("Fatal [%s] : %s\n", L"Fatal [%s] : %s\n"), _name.data(), msg.data());
                    break;
                default:
                    break;
                }
            }
            template<typename T>
            void Log(const T* msg, const level_e nLevel) const
            {
                std::basic_string<T, std::char_traits<T>, std::allocator<T>> _msg = msg;
                Log(_msg, nLevel);
            }

            /*
            *@brief 打印Debug信息
            *@param _func 宏 __FUNCTION__ 函数名
            *@param _file 宏 __FILE__ 文件名
            *@param _line 宏 __LINE__ 行数
            */
            template<typename T>
            void Log(T msg, const level_e nLevel, const char* _func, const char* _file, const int _line)const
            {
                if (nLevel < m_level || nLevel == level_e::Off) { return; }
                T func = tvalue<T>(_func, codecvt::ansi_to_unicode(_func));
                T file = tvalue<T>(_file, codecvt::ansi_to_unicode(_file));

                size_t n = file.find_last_of('\\');
                n = file.find_last_of('\\', n - 1);
                n = file.find_last_of('\\', n - 1);
                if (n == static_cast<size_t>(-1)) { n = 0; }
                T file_fmt = T(file.begin() + n, file.end());

                T _name = tvalue<T>(m_name, codecvt::ansi_to_unicode(m_name));
                if (m_name == "") { _name = tvalue<T>(" ", L" "); }

                switch (nLevel)
                {
                case level_e::Trace:
                    OutputDebugStringEx()(tvalue<T>("Trace [%s] : %s   [...%s(%d) %s]\n", L"Trace [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Debug:
                    OutputDebugStringEx()(tvalue<T>("Debug [%s] : %s   [...%s(%d) %s]\n", L"Debug [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Info:
                    OutputDebugStringEx()(tvalue<T>("Info  [%s] : %s   [...%s(%d) %s]\n", L"Info  [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Warn:
                    OutputDebugStringEx()(tvalue<T>("Warn  [%s] : %s   [...%s(%d) %s]\n", L"Warn  [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Error:
                    OutputDebugStringEx()(tvalue<T>("Error [%s] : %s   [...%s(%d) %s]\n", L"Error [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                case level_e::Fatal:
                    OutputDebugStringEx()(tvalue<T>("Fatal [%s] : %s   [...%s(%d) %s]\n", L"Fatal [%s] : %s   [...%s(%d) %s]\n"),
                        _name.data(), msg.data(), file_fmt.data(), _line, func.data());
                    break;
                default:
                    break;
                }
            }
            template<typename T>
            void Log(const T* msg, const level_e nLevel, const char* _func, const char* _file, const int _line)const
            {
                std::basic_string<T, std::char_traits<T>, std::allocator<T>> _msg = msg;
                Log(_msg, nLevel, _func, _file, _line);
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
            void Trace(T msg, const char* _func, const char* _file, const int _line) const { Log(std::move(msg), level_e::Trace, _func, _file, _line); }
            template<typename T>
            void Debug(T msg, const char* _func, const char* _file, const int _line) const { Log(std::move(msg), level_e::Debug, _func, _file, _line); }
            template<typename T>
            void Info(T msg, const char* _func, const char* _file, const int _line) const { Log(std::move(msg), level_e::Info, _func, _file, _line); }
            template<typename T>
            void Warn(T msg, const char* _func, const char* _file, const int _line) const { Log(std::move(msg), level_e::Warn, _func, _file, _line); }
            template<typename T>
            void Error(T msg, const char* _func, const char* _file, const int _line) const { Log(std::move(msg), level_e::Error, _func, _file, _line); }
            template<typename T>
            void Fatal(T msg, const char* _func, const char* _file, const int _line) const { Log(std::move(msg), level_e::Fatal, _func, _file, _line); }

        private:
            std::string m_name;
            level_e m_level;
        };

        /// @}
    } // namespace debuglog

    /*< 全局静态logger对象 管理默认记录级别 */
    static std::unique_ptr<debuglog::logger> g_logger(new debuglog::logger(_T("G"), level_e::Trace));

    namespace debuglog {
        /// @addtogroup debuglog
        /// @{
        /**
         @brief 封装debugoutput log流, 需要一个结束符将流内容输入到debugoutput
        */
        class dlog_ostream :public std::ostream
        {
        private:
            level_e _level;
            std::mutex _log_mu;      // 多线程log时保证以消息为单位
            std::ostringstream _oss;
        public:

            level_e& get_level() {
                return _level;
            };

            std::mutex& get_log_mu() {
                return _log_mu;
            };

            std::ostringstream& get_oss() {
                return _oss;
            };

        public:

            dlog_ostream(level_e level) :std::ostream(0), _level(level)
            {
            }

            template<typename T>
            dlog_ostream& operator<<(T str)
            {
                _oss << str;
                return (*this);
            }

            dlog_ostream& operator<<(std::wstring str)
            {
                _oss << codecvt::unicode_to_ansi(str);
                return (*this);
            }

            dlog_ostream& operator<<(const wchar_t * str)
            {
                _oss << codecvt::unicode_to_ansi(str);
                return (*this);
            }

            template<typename T, typename = typename std::enable_if_t<std::is_same_v<T, dlog_ostream&>>>
            dlog_ostream& operator<<(dlog_ostream& str)
            {
                _oss << str._oss.str();
                return (*this);
            }

            template<class _Elem = char, class _Traits = std::char_traits<char>>
            dlog_ostream& operator<<(std::basic_ostream<_Elem, _Traits>& str)
            {
                _oss << str._oss.str();
                return (*this);
            }

            template<class _Elem = char, class _Traits = std::char_traits<char>>
            dlog_ostream& operator<<(std::basic_ostream<_Elem, _Traits>& (__cdecl *_Pfn)(std::basic_ostream<_Elem, _Traits>&))
            {
                (*_Pfn)(this->_oss);
                return (*this);
            }

            dlog_ostream& operator<<(dlog_ostream& (__cdecl * Pfn)(dlog_ostream&))
            {
                return (*Pfn)(*this);
            }

            dlog_ostream& operator<<(std::function<dlog_ostream&(dlog_ostream&)> Fn)
            {
                return Fn(*this);
            }

            /**
             @brief 打印输出并清空oss
             @note 由于当前logger为固定format格式,以消息为单位 所以stream需要一个结束符
             @see logger
            */
            static dlog_ostream& dend(dlog_ostream& dlog)
            {
                dlog._log_mu.lock();
                common::g_logger->Log(std::move(dlog._oss.str()), dlog._level);
                dlog._oss.str("");
                dlog._log_mu.unlock();
                return dlog;
            }
            // @override
            static dlog_ostream& dend_(dlog_ostream& dlog, const char* _func, const char* _file, const int _line)
            {
                dlog._log_mu.lock();
                common::g_logger->Log(std::move(dlog._oss.str()), dlog._level, _func, _file, _line);
                dlog._oss.str("");
                dlog._log_mu.unlock();
                return dlog;
            }
        };

        template<typename T>
        dlog_ostream&& operator<<(dlog_ostream&& os, const T& val)
        {
            os << val;
            return std::move(os);
        }

        /// @}
    } // namespace debuglog

    template<typename T>
    inline void LOGT(T msg) { g_logger->Trace(std::move(msg)); };
    template<typename T>
    inline void LOGD(T msg) { g_logger->Debug(std::move(msg)); };
    template<typename T>
    inline void LOGI(T msg) { g_logger->Info(std::move(msg)); };
    template<typename T>
    inline void LOGW(T msg) { g_logger->Warn(std::move(msg)); };
    template<typename T>
    inline void LOGE(T msg) { g_logger->Error(std::move(msg)); };
    template<typename T>
    inline void LOGF(T msg) { g_logger->Fatal(std::move(msg)); };

    template<typename T>
    inline void LOGT(T msg, const char* _func, const char* _file, const int _line) { g_logger->Trace(std::move(msg), _func, _file, _line); };
    template<typename T>
    inline void LOGD(T msg, const char* _func, const char* _file, const int _line) { g_logger->Debug(std::move(msg), _func, _file, _line); };
    template<typename T>
    inline void LOGI(T msg, const char* _func, const char* _file, const int _line) { g_logger->Info(std::move(msg), _func, _file, _line); };
    template<typename T>
    inline void LOGW(T msg, const char* _func, const char* _file, const int _line) { g_logger->Warn(std::move(msg), _func, _file, _line); };
    template<typename T>
    inline void LOGE(T msg, const char* _func, const char* _file, const int _line) { g_logger->Error(std::move(msg), _func, _file, _line); };
    template<typename T>
    inline void LOGF(T msg, const char* _func, const char* _file, const int _line) { g_logger->Fatal(std::move(msg), _func, _file, _line); };

// debug log ostream 可调用模式 () ,下划线版本输出位置

#define LOGT_(msg) common::LOGT(msg, __FUNCTION__, __FILE__, __LINE__)
#define LOGD_(msg) common::LOGD(msg, __FUNCTION__, __FILE__, __LINE__)
#define LOGI_(msg) common::LOGI(msg, __FUNCTION__, __FILE__, __LINE__)
#define LOGW_(msg) common::LOGW(msg, __FUNCTION__, __FILE__, __LINE__)
#define LOGE_(msg) common::LOGE(msg, __FUNCTION__, __FILE__, __LINE__)
#define LOGF_(msg) common::LOGF(msg, __FUNCTION__, __FILE__, __LINE__)

// debug log ostream 流输出模式 <<

#define DLOGT common::debuglog::dlog_ostream(common::level_e::Trace)
#define DLOGD common::debuglog::dlog_ostream(common::level_e::Debug)
#define DLOGI common::debuglog::dlog_ostream(common::level_e::Info)
#define DLOGW common::debuglog::dlog_ostream(common::level_e::Warn)
#define DLOGE common::debuglog::dlog_ostream(common::level_e::Error)
#define DLOGF common::debuglog::dlog_ostream(common::level_e::Fatal)
#define DEND  common::debuglog::dlog_ostream::dend
#define DEND_ std::function<common::debuglog::dlog_ostream&(common::debuglog::dlog_ostream&)>( \
    std::bind(common::debuglog::dlog_ostream::dend_, std::placeholders::_1,__FUNCTION__, __FILE__, __LINE__)); // 拖尾"_"表示打印位置

    /// @}
} // namespace common

#endif // _COMMON_DEBUGLOG_HPP_

#ifdef _MSC_VER
#pragma warning(pop)
#endif