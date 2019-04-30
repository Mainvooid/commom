#include <common.h>

using namespace common;

/**
解析命令行获取解析字符串数组.
*/
void cmdline_parser(int argc, char* argv[], std::string result[]) {
	// 创建命令行解析器
	cmdline::parser parser;

	// 添加参数
	// 长名称,短名称(\0表示没有短名称),参数描写,是否必须,默认值.
	// cmdline::range()限制范围,cmdline::oneof<>()限制可选值.
	// 通过调用不带类型的add方法,定义bool值(通过调用exsit()方法来推断).
	parser.add<std::string>("input_fname", 'i', "input file name.", true, "test.in");
	parser.add<std::string>("output_fname", 'o', "output file name.", true, "test.out");

	// 执行解析
	parser.parse_check(argc, argv);

	// 获取输入的参数值
	result[0] = parser.get<std::string>("input_fname");
	result[1] = parser.get<std::string>("output_fname");
}

int main(int argc, char* argv[])
{
	std::string param[2];
	cmdline_parser(argc, argv, param);

	std::string input_fname = param[0];
	std::string output_fname = param[1];
}