/*
@brief  a simple command line parser
@author guobao.v@gmail.com
*/
#ifndef _COMMON_CMDLINE_HPP_
#define _COMMON_CMDLINE_HPP_

#include <common/precomm.hpp>
#include <algorithm>
#include <map>

#ifdef __GNUC__
#include <cxxabi.h>
#endif

namespace common {
    namespace cmdline {
        static const std::string _TAG = "cmdline";

        namespace detail {
            /**
            *@brief 获取类型
            */
            static inline std::string demangle(const std::string &name) noexcept
            {
#ifdef _MSC_VER
                return name;
#elif defined(__GNUC__)
                int status = 0;
                char *p = abi::__cxa_demangle(name.c_str(), 0, 0, &status);
                std::string ret(p);
                free(p);
                return ret;
#else
#error unexpected c complier (msc/gcc), Need to implement this method for demangle
#endif
            }

            /**
            *@brief 获取类型
            */
            template <typename T>
            std::string readable_typename() noexcept
            {
                return demangle(typeid(T).name());
            }
            template <>
            inline std::string readable_typename<std::string>() noexcept { return "string"; }
            template <>
            inline std::string readable_typename<std::wstring>() noexcept { return "wstring"; }

            /**
            *@brief T -> string
            */
            template <typename T>
            std::string default_value(T def) noexcept
            {
                return convert_to_string<char>(def);
            }
        } // detail --------------------------------------------------

        /**
        *@brief 模块异常类
        */
        class[[deprecated("unnecessary")]]cmdline_error : public std::exception
        {
        public:
            cmdline_error(const std::string msg) : m_msg(std::move(msg)) {}
            ~cmdline_error() {}
            const char *what() const { return m_msg.c_str(); }
        private:
            std::string m_msg;
        };

        /**
        *@brief string -> T
        */
        template <typename T>
        class default_reader
        {
        public:
            T operator()(const std::string &str) noexcept(false)
            {
                return convert_from_string<T>(str);
            }
        };

        /**
        *@brief 参数范围检查器
        */
        template <typename T>
        class range_reader
        {
        public:
            range_reader(const T &low, const T &high) : m_low(low), m_high(high) {}
            T operator()(const std::string &s) const noexcept(false)
            {
                T ret = default_reader<T>()(s);
                if (!(ret >= m_low && ret <= m_high)) {
                    std::ostringstream msg;
                    msg << _TAG << "..range error";
                    throw std::range_error(msg.str());
                }
                return ret;
            }
        private:
            T m_low, m_high;
        };

        /**
        *@brief 返回一个参数范围检查器
        */
        template <typename T>
        range_reader<T> range(const T &low, const T &high) noexcept
        {
            return range_reader<T>(low, high);
        }

        /**
        *@brief 可选值检查器
        */
        template <typename T>
        class oneof_reader
        {
        public:
            oneof_reader() {}
            oneof_reader(const std::initializer_list<T> &list) noexcept
            {
                for (T item : list) {
                    m_values.push_back(item);
                }
            }

            template <typename ...Values>
            oneof_reader(const T& v, const Values&...vs) noexcept
            {
                add(v, vs...);
            }

            T operator=(const std::initializer_list<T> &list) noexcept
            {
                for (T item : list) {
                    m_values.push_back(item);
                }
            };

            T operator()(const std::string &s) noexcept(false)
            {
                T ret = default_reader<T>()(s);
                if (std::find(m_values.begin(), m_values.end(), ret) == m_values.end()) {
                    std::ostringstream msg;
                    msg << _TAG << "..oneof error";
                    throw std::invalid_argument(msg.str());
                }
                return ret;
            }

            template <typename ...Values>
            void add(const T& v, const Values&...vs) noexcept
            {
                m_values.push_back(v);
                add(vs...);
            }

        private:
            void add(const T& v) noexcept
            {
                m_values.push_back(v);
            }

        private:
            std::vector<T> m_values;
        };

        /**
        *@brief 返回一个可选值检查器
        */
        template <typename T, typename ...Values>
        oneof_reader<T> oneof(const T& a1, const Values&... a2) noexcept
        {
            return oneof_reader<T>(a1, a2...);
        }

        /**
        *@brief 返回一个可选值检查器
        */
        template <typename T>
        oneof_reader<T> oneof(const std::initializer_list<T> &list) noexcept
        {
            return oneof_reader<T>(list);
        }

        /**
        *@brief 命令行解析类
        */
        class parser
        {
        public:
            parser() {}
            ~parser()
            {
                for (std::map<std::string, option_base*>::iterator p = options.begin(); p != options.end(); p++) {
                    delete_s(p->second);
                }
            }

            /**
            *@brief 添加指定类型的参数
            *@param name       长名称
            *@param short_name 短名称(\0:表示没有短名称)
            *@param desc       描述
            */
            void add(const std::string &name, char short_name = 0, const std::string &desc = "") noexcept(false)
            {
                if (options.count(name)) {
                    std::ostringstream msg;
                    msg << _TAG << "..multiple definition:" << name;
                    throw std::invalid_argument(msg.str());
                }
                options[name] = new option_without_value(name, short_name, desc);
                ordered.push_back(options[name]);
            }

            /**
            *@brief 添加指定类型的参数
            *@param name       长名称
            *@param short_name 短名称(\0:表示没有短名称)
            *@param desc       描述
            *@param need       是否必需(可选)
            *@param def        默认值(可选,当不必需时使用)
            */
            template <typename T>
            void add(const std::string &name, char short_name = 0, const std::string &desc = "",
                bool need = true, const T def = T()) noexcept(false)
            {
                add(name, short_name, desc, need, def, default_reader<T>());
            }

            /**
            *@brief 添加指定类型的参数
            *@param name       长名称
            *@param short_name 短名称(\0:表示没有短名称)
            *@param desc       描述
            *@param need       是否必需(可选)
            *@param def        默认值(可选,当不必需时使用)
            *@param reader     解析类型
            */
            template <class T, class F>
            void add(const std::string &name, char short_name = 0, const std::string &desc = "",
                bool need = true, const T def = T(), F reader = F()) noexcept(false)
            {
                if (options.count(name)) {
                    std::ostringstream msg;
                    msg << _TAG << "..multiple definition:" << name;
                    throw std::invalid_argument(msg.str());
                }
                options[name] = new option_with_value_with_reader<T, F>(name, short_name, need, def, desc, reader);
                ordered.push_back(options[name]);
            }

            /**
            *@brief usage尾部添加说明(如果需要解析未指定参数)
            *@param f 补充说明
            */
            void footer(std::string f) noexcept { ftr = std::move(f); }

            /**
            *@brief 设置usage程序名,默认由argv[0]确定
            *@param name usage程序名
            */
            void set_program_name(std::string name) noexcept { prog_name = std::move(name); }

            /**
            *@brief 判断bool参数是否被指定
            *@param name bool参数名
            */
            bool exist(const std::string &name) const noexcept(false)
            {
                if (options.count(name) == 0) {
                    std::ostringstream msg;
                    msg << _TAG << "..there is no flag: --" << name;
                    throw std::invalid_argument(msg.str());
                }
                return options.find(name)->second->has_set();
            }

            /**
            *@brief 获取参数的值
            *@param name 参数名
            *@return 返回相应类型参数值
            */
            template <class T>
            const T &get(const std::string &name) const noexcept(false)
            {
                if (options.count(name) == 0) {
                    std::ostringstream msg;
                    msg << _TAG << "..there is no flag: --" << name;
                    throw std::invalid_argument(msg.str());
                }
                const option_with_value<T> *p = dynamic_cast<const option_with_value<T>*>(options.find(name)->second);
                if (p == nullptr) {
                    std::ostringstream msg;
                    msg << _TAG << "..type mismatch flag '" << name << "'";
                    throw std::invalid_argument(msg.str());
                }
                return p->get();
            }

            /**
            *@brief 获取未指定的参数的值
            */
            const std::vector<std::string> &rest() const noexcept { return others; }

            /**
            *@brief 解析一行命令
            *@param arg 参数
            *@return 是否解析成功
            */
            bool parse(const std::string &arg) noexcept
            {
                std::vector<std::string> args;
                std::string buf;
                bool in_quote = false;//是否有""
                for (std::string::size_type i = 0; i < arg.length(); i++) {
                    if (arg[i] == '\"') {
                        in_quote = !in_quote;
                        continue;
                    }
                    if (arg[i] == ' ' && !in_quote) {
                        args.push_back(buf);
                        buf = "";
                        continue;
                    }
                    if (arg[i] == '\\') { //跳过'\'
                        i++;
                        if (i >= arg.length()) {
                            errors.push_back("unexpected occurrence of '\\' at end of string");
                            return false;
                        }
                    }
                    buf += arg[i];
                }
                if (in_quote) {
                    errors.push_back("quote is not closed");
                    return false;
                }
                if (buf.length() > 0) {
                    args.push_back(buf);
                }
                for (size_t i = 0; i < args.size(); i++) {
                    std::cout << "\"" << args[i] << "\"" << std::endl;
                }
                return parse(args);
            }

            /**
            *@brief 解析参数数组
            *@param args 参数数组
            *@return 是否解析成功
            */
            bool parse(const std::vector<std::string> &args) noexcept
            {
                int argc = static_cast<int>(args.size());
                std::vector<const char*> argv(argc);
                for (int i = 0; i < argc; i++) {
                    argv[i] = args[i].c_str();
                }
                return parse(argc, &argv[0]);
            }

            /**
            *@brief 解析参数数组
            *@param argc 参数数量(+程序名)
            *@param argv 参数值([0]为程序名)
            *@return 是否解析成功
            */
            bool parse(int argc, const char * const argv[]) noexcept
            {
                errors.clear();
                others.clear();
                if (argc < 1) {
                    errors.push_back("argument number must be bigger than 0");
                    return false;
                }
                if (prog_name == "") { prog_name = argv[0]; }

                std::map<char, std::string> lookup;
                for (std::map<std::string, option_base*>::iterator p = options.begin(); p != options.end(); p++) {
                    if (p->first.length() == 0) {
                        continue;//key不存在
                    }
                    char initial = p->second->short_name();
                    if (initial) {
                        if (lookup.count(initial) > 0) {
                            lookup[initial] = "";
                            errors.push_back(std::string("short option '") + initial + "' is ambiguous");
                            return false;
                        }
                        else {
                            lookup[initial] = p->first;
                        }
                    }
                }

                for (int i = 1; i < argc; i++) {
                    if (strncmp(argv[i], "--", 2) == 0) {//长名称
                        const char *p = strchr(argv[i] + 2, '=');
                        if (p) {
                            std::string name(argv[i] + 2, p);
                            std::string val(p + 1);
                            set_option(name, val);
                        }
                        else {
                            std::string name(argv[i] + 2);
                            if (options.count(name) == 0) {
                                errors.push_back("undefined option: --" + name);
                                continue;
                            }
                            if (options[name]->has_value()) {
                                if (i + 1 >= argc) {
                                    errors.push_back("option needs value: --" + name);
                                    continue;
                                }
                                else {
                                    i++;
                                    set_option(name, argv[i]);
                                }
                            }
                            else {
                                set_option(name); //bool
                            }
                        }
                    }
                    else if (strncmp(argv[i], "-", 1) == 0) {//短名称
                        if (!argv[i][1]) {//若'-'后无值
                            continue;
                        }
                        char last = argv[i][1];
                        for (int j = 2; argv[i][j]; j++) {
                            last = argv[i][j];
                            if (lookup.count(argv[i][j - 1]) == 0) {
                                errors.push_back(std::string("undefined short option: -") + argv[i][j - 1]);
                                continue;
                            }
                            if (lookup[argv[i][j - 1]] == "") {
                                errors.push_back(std::string("ambiguous short option: -") + argv[i][j - 1]);
                                continue;
                            }
                            set_option(lookup[argv[i][j - 1]]);
                        }

                        if (lookup.count(last) == 0) {
                            errors.push_back(std::string("undefined short option: -") + last);
                            continue;
                        }
                        if (lookup[last] == "") {
                            errors.push_back(std::string("ambiguous short option: -") + last);
                            continue;
                        }

                        if (i + 1 < argc && options[lookup[last]]->has_value()) {
                            set_option(lookup[last], argv[i + 1]);
                            i++;
                        }
                        else {
                            set_option(lookup[last]);
                        }
                    }
                    else {
                        others.push_back(argv[i]);
                    }
                }

                for (std::map<std::string, option_base*>::iterator p = options.begin(); p != options.end(); p++) {
                    if (!p->second->valid()) {
                        errors.push_back("need option: --" + std::string(p->first));
                    }
                }
                return errors.size() == 0;
            }

            /**
            *@brief 包装parse并做检查
            *@param arg 一行命令
            */
            void parse_check(const std::string &arg) noexcept(false)
            {
                if (!options.count("help")) {
                    add("help", '?', "print this message");
                }
                check(0, parse(arg));
            }

            /**
            *@brief 包装parse并做检查
            *@param args 参数数组
            */
            void parse_check(const std::vector<std::string> &args) noexcept(false)
            {
                if (!options.count("help")) {
                    add("help", '?', "print this message");
                }
                check(args.size(), parse(args));
            }

            /**
            *@brief 运行解析器(包装parse并做检查)
            *@param argc 参数数量(+程序名)
            *@param argv 参数值([0]为程序名)
            *@note 仅当命令行参数有效时才返回
                   如果参数无效，解析器输出错误消息然后退出程序
                   如果指定了help flag('-help'或'-?')或空命令，则解析器输出用法消息然后退出程序
            */
            void parse_check(int argc, char *argv[]) noexcept(false)
            {
                if (!options.count("help")) {
                    add("help", '?', "print this message");
                }
                check(argc, parse(argc, argv));
            }

            /**
            *@brief 返回第一条错误消息
            */
            std::string error() const { return errors.size() > 0 ? errors[0] : ""; }

            /**
            *@brief 返回所有错误消息
            */
            std::string error_full() const noexcept
            {
                std::ostringstream oss;
                for (size_t i = 0; i < errors.size(); i++) {
                    oss << errors[i] << std::endl;
                }
                return oss.str();
            }

            /**
            *@brief 返回使用方法说明
            */
            std::string usage() const noexcept
            {
                std::ostringstream oss;
                oss << "usage: " << prog_name << " ";
                for (size_t i = 0; i < ordered.size(); i++) {
                    if (ordered[i]->must()) {
                        oss << ordered[i]->short_description() << " ";
                    }
                }

                oss << "[options] ... " << ftr << std::endl;
                oss << "options:" << std::endl;

                size_t max_width = 0;
                for (size_t i = 0; i < ordered.size(); i++) {
                    max_width = std::max(max_width, ordered[i]->name().length());
                }

                for (size_t i = 0; i < ordered.size(); i++) {
                    if (ordered[i]->short_name()) {
                        oss << "  -" << ordered[i]->short_name() << ", ";
                    }
                    else {
                        oss << "      ";
                    }

                    oss << "--" << ordered[i]->name();
                    for (size_t j = ordered[i]->name().length(); j < max_width + 4; j++) {
                        oss << ' ';
                    }
                    oss << ordered[i]->description() << std::endl;
                }
                return oss.str();
            }

        private:
            /**
            *@brief parse检查
            *@param argc 参数数量
            *@param ok 是否解析成功
            */
            void check(size_t argc, bool ok) noexcept
            {
                if ((argc == 1 && !ok) || exist("help")) {
                    std::cerr << usage();
                    exit(0);
                }
                if (!ok) {
                    std::cerr << error() << std::endl << usage();
                    exit(1);
                }
            }

            /**
            *@brief 设置参数选项
            *@param name 参数名
            */
            void set_option(const std::string &name) noexcept
            {
                if (options.count(name) == 0) {
                    errors.push_back("undefined option: --" + name);
                    return;
                }
                if (!options[name]->set()) {
                    errors.push_back("option needs value: --" + name);
                    return;
                }
            }

            /**
            *@brief 设置参数选项
            *@param name 参数名
            *@param value 参数值
            */
            void set_option(const std::string &name, const std::string &value) noexcept {
                if (options.count(name) == 0) {
                    errors.push_back("undefined option: --" + name);
                    return;
                }
                if (!options[name]->set(value)) {
                    errors.push_back("option value is invalid: --" + name + "=" + value);
                    return;
                }
            }

            /**
            *@brief 参数选项基础接口类
            */
            class option_base {
            public:
                virtual ~option_base() {}

                virtual bool has_value() const = 0;
                virtual bool set() = 0;
                virtual bool set(const std::string &value) = 0;
                virtual bool has_set() const = 0;
                virtual bool valid() const = 0;
                virtual bool must() const = 0;

                virtual const std::string &name() const = 0;
                virtual char short_name() const = 0;
                virtual const std::string &description() const = 0;
                virtual std::string short_description() const = 0;
            };

            /**
            *@brief 参数选项派生类(无值参数:bool)
            */
            class option_without_value : public option_base {
            public:
                option_without_value(const std::string &name, char short_name, const std::string &desc)
                    :m_name(name), m_short_name(short_name), m_desc(desc), m_has(false) {}
                ~option_without_value() {}

                bool has_value() const noexcept { return false; }
                bool set() noexcept { m_has = true; return true; }
                bool set(const std::string &) noexcept { return false; }
                bool has_set() const noexcept { return m_has; }
                bool valid() const noexcept { return true; }
                bool must() const noexcept { return false; }
                const std::string &name() const noexcept { return m_name; }
                char short_name() const noexcept { return m_short_name; }
                const std::string &description() const noexcept { return m_desc; }
                std::string short_description() const noexcept { return "--" + m_name; }

            private:
                std::string m_name;
                char m_short_name;
                std::string m_desc;
                bool m_has;
            };

            /**
            *@brief 参数选项派生类(有值参数)
            */
            template <class T>
            class option_with_value : public option_base {
            public:
                /**
                *@brief 参数选项派生类(有值参数)
                *@param name       长名称
                *@param short_name 短名称
                *@param need       是否必需
                *@param def        默认值
                *@param desc       描述
                */
                option_with_value(const std::string &name, char short_name, bool need, const T &def, const std::string &desc)
                    : m_name(name), m_short_name(short_name), m_need(need), m_has(false), m_def(def), m_actual(def)
                {
                    this->desc = full_description(desc);
                }
                ~option_with_value() {}

                const T &get() const { return m_actual; }
                bool has_value() const { return true; }
                bool set() { return false; }

                bool set(const std::string &value) noexcept
                {
                    try {
                        m_actual = read(value);
                        m_has = true;
                    }
                    catch (const std::exception &) {
                        return false;
                    }
                    return true;
                }

                bool has_set() const { return m_has; }
                bool valid() const { return (m_need && !m_has) ? false : true; }
                bool must() const { return m_need; }
                const std::string &name() const { return m_name; }
                char short_name() const { return m_short_name; }
                const std::string &description() const { return desc; }

                std::string short_description() const noexcept
                {
                    return "--" + m_name + "=" + detail::readable_typename<T>();
                }

            protected:
                std::string full_description(const std::string &desc) noexcept
                {
                    return
                        desc + " (" + detail::readable_typename<T>() +
                        (m_need ? "" : " [=" + detail::default_value<T>(m_def) + "]")
                        + ")";
                }

                virtual T read(const std::string &s) = 0;

            protected:
                std::string m_name;
                char m_short_name;
                bool m_need;
                std::string desc;
                bool m_has;
                T m_def;//默认值
                T m_actual;//实际值
            };

            /**
            *@brief 有值参数选项派生类
            */
            template <class T, class F>
            class option_with_value_with_reader : public option_with_value<T>
            {
            public:
                /**
                *@brief 有值参数选项派生类
                *@param name       长名称
                *@param short_name 短名称
                *@param need       是否必需
                *@param def        默认值
                *@param desc       描述
                *@param reader     可读包装类型string->T
                */
                option_with_value_with_reader(const std::string &name,
                    char short_name,
                    bool need,
                    const T def,
                    const std::string &desc,
                    F reader)
                    : option_with_value<T>(name, short_name, need, def, desc), reader(reader) {}

            private:
                //string -> T
                T read(const std::string &s) noexcept { return reader(s); }

            private:
                F reader;
            };

        private:
            std::map<std::string, option_base*> options;//参数选项map(长名称,一个选项)
            std::vector<option_base*> ordered;          //有序的参数选项(add时push)
            std::string ftr;                            //usage尾部添加说明
            std::string prog_name;                      //程序名
            std::vector<std::string> others;            //其他为指定参数
            std::vector<std::string> errors;            //错误消息
        };

    } // cmdline

} // common

#endif // _COMMON_CMDLINE_HPP_