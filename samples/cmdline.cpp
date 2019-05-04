/*
@brief usage sample for cmdline.h.
@author guobao.v@gmail.com
*/

#include <common/cmdline.h>
#include <iostream>

using namespace std;
using namespace common;
using namespace common::cmdline;

/**
*@brief 获取指定参数(用于终止递归)
*@parser 命令行解析器
*@arg_names 参数长名称数组
*@arg 获取到的参数值
*/
template <typename T>
void getArgs(const parser &parser, vector<string> &arg_name, T &arg)
{
	if (is_same<T, bool>::value)
	{
		arg = parser.exist(arg_name[0]);
	}
	else
	{
		arg = parser.get<T>(arg_name[0]);
	}
	arg_name.erase(arg_name.begin());
}

/**
*@brief 获取指定参数
*@parser 命令行解析器
*@arg_names 参数长名称数组
*@arg 参数包第一个参数的绑定
*@rest 参数包,用于接收参数值
*/
template<typename T, typename...Args>
void getArgs(const parser &parser, vector<string> &arg_names, T &arg, Args&...args)
{
	if (is_same<T, bool>::value)
	{
		arg = parser.exist(arg_names[0]);
	}
	else
	{
		//const parser::option_with_value<T> *p = dynamic_cast<const parser::option_with_value<T>*>("aaa");
		//arg=p->get();
		arg = parser.get<T>(arg_names[0]);
	}
	arg_names.erase(arg_names.begin());

	getArgs(parser, arg_names, args...);
}

/**
解析命令行获取解析字符串数组.
*/
template<typename...Args>
void cmdline_parser(int argc, char* argv[], Args &...args) {
	// 创建命令行解析器
	parser parser;

	// 添加参数
	vector<string> name = { "input_fname", "output_fname", "port", "type","zip" };
	parser.add<string>(name[0], 'i', "input file name.", true, "test.in");
	parser.add<string>(name[1], 'o', "output file name.", true, "test.out");

	// cmdline::range()限制范围
	parser.add<int>(name[2], 'p', "port number", false, 80, range(1, 65535));

	// cmdline::oneof<>()限制可选值.
	parser.add<string>(name[3], 't', "file type", false, "mp4", oneof<string>("h264", "mp4", "avi"));

	// 通过调用不带类型的add方法,定义bool值(通过调用exsit()方法来推断).
	parser.add(name[4], '\0', "gzip when transfer");

	//usage尾部添加说明
	parser.footer("filename ...");

	//设置usage程序名,默认由argv[0]确定
	//parser.set_program_name("common");

	//运行解析器
	parser.parse_check(argc, argv);

	// 获取指定的输入参数值
	getArgs(parser, name, args...);

	// 其余参数,返回字符串数组,通常用于返回指定的文件名
	//vector<string> rest = parser.rest();
}

int main(int argc, char* argv[])
{
	argc = 10;
	argv = (char **)malloc(sizeof(char*) * argc);
	argv[0] = const_cast<char*>("common.exe");
	argv[1] = const_cast<char*>("-i");
	argv[2] = const_cast<char*>("in");
	argv[3] = const_cast<char*>("-o");
	argv[4] = const_cast<char*>("out");
	argv[5] = const_cast<char*>("-p");
	argv[6] = const_cast<char*>("8080");
	argv[7] = const_cast<char*>("-t");
	argv[8] = const_cast<char*>("mp4");
	argv[9] = const_cast<char*>("--zip");

	string input_fname;
	string output_fname;
	int port;
	string type;
	bool isZip;
	cmdline_parser(argc, argv, input_fname, output_fname, port, type, isZip);

	cout << "input_fname:" << input_fname << endl;
	cout << "output_fname:" << output_fname << endl;
	cout << "port:" << port << endl;
	cout << "type:" << type << endl;
	cout << "zip:" << isZip << endl;
	free_s(argv);
}
/*
>./common
usage: common.exe --input_fname=string --output_fname=string [options] ... filename ...
options:
  -i, --input_fname     input file name. (string)
  -o, --output_fname    output file name. (string)
  -p, --port            port number (int [=80])
  -t, --type            file type (string [=mp4])
	  --zip             gzip when transfer
  -?, --help            print this message

>./common -i "in" -o "out" -p 8080 -t "mp4" --zip "test.exe"
input_fname:in
output_fname:out
port:8080
type:mp4
zip:1
*/