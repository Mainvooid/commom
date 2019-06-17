# commom
common library only include header files.

It has only been tested on Windows.

---

## Tree

```cpp
│  code.snippet     //文件代码模板(vs->工具->代码片段管理中导入)
│  common.hpp       //主头文件(使用时include本文件)
│  common_all.hpp   //统一的主头文件(暂未生成)
├─common            //分头文件目录
│  │  cmdline.hpp      //命令行解析
│  │  codecvt.hpp      //字符编码转换
│  │  cuda.hpp         //cuda辅助(包含与opencv和directx的互操作)
│  │  debuglog.hpp     //windows调试日志
│  │  opencl.hpp       //opencl辅助
│  │  opencv.hpp       //opencv辅助
│  │  precomm.hpp      //公共辅助
│  │  windows.hpp      //windows辅助(包含directx辅助)
│  └─cuda           //cuda设备函数目录
│     │  texture_reference.cu    
│     └─ texture_reference.cuh
├─docs             //文档目录
├─samples          //使用样例目录
│  └─data             //测试数据 
└─tests            //单元测试目录
```

---

## Macro

默认关闭库/宏支持
- `HAVE_OPENCL`      //基于OpenCL 1.2
- `HAVE_OPENCV `     //基于OpenCV 4.0 with contrib
- `HAVE_DIRECTX`     //基于Microsoft DirectX SDK (June 2010)
- `HAVE_CUDA`        //基于CUDA 10.0
- `HAVE_CUDA_DEVICE` // 本项目cuda目录下的.cu文件添加到工程后可以开启本宏,宏详细说明见[common/cuda/README.md](common/cuda/README.md)

---

## Code style and code specification

**同一文件内应统一风格.**

### 命名

缩进应使用4空格而非制表符,语句后不应尾随空格.

简化匈牙利命名法:
前缀|类型
---|---
g_ |全局变量
m_ |类成员变量
s_ |静态变量
c_ |常量
p_ |指针变量

所有前缀或后缀应该写在一起用一个下划线隔开:
前后缀|类型
---|---
mp_  |成员指针变量
mcp_ |成员常量指针
mpc_ |成员指针常量
_t   |结构体类型
_fn  |函数指针类型
_e   |枚举类型

函数内临时的同名变量可以加个前缀`_`代表临时.
变量/文件命名全小写+下划线

若驼峰式命名法
类名用大驼峰,变量第一个词是动词则小驼峰addOption().

### 文件

源文件内的头文件包含顺序应从最特殊到一般，如：
```cpp
#include "通用头文件"
#include "源文件同名头文件"
#include "本模块其他头文件"
#include "自定义工具头文件"
#include "第三方头文件"
#include "平台相关头文件"
#include "C++库头文件"
#include "C库头文件"
```

模块应使用命名空间`namespace{}`包含.

#### HPP文件要注意的问题

所有HPP文件使用宏避免重复包含.
```cpp
#ifndef _COMMON_PRECOMM_HPP_
#define _COMMON_PRECOMM_HPP_
#endif
```

HPP文件中可以使用using引用依赖,不应该使用using namespace污染命名空间.

函数的重定义问题:
- 将全局函数封装为类的静态方法
- 通过冗余的模板参数变成模板函数
```cpp
template<bool flag=false>
```
- static修饰

### Doxygen文档

#### 注释

```
/**
 *  多行注释
 */
 
/**单行注释*/ 或 ///

/**<同行注释 */ 或 ///< (Doxygen认为注释是修饰接下来的程序代码的)
```

#### 文件信息

```
@file      文件名
@author    作者名
@version   版本号
@todo      待办事项
@date      日期时间
@section   章节标题 e.g. [@section LICENSE 版权许可] [@section DESCRIPTION 描述]
```
#### 模块信息

```
@defgroup   定义模块                         模块名(英文) 显示名    @{ 类/函数/变量/宏/... @}
@ingroup    作为指定名的模块的子模块          模块名(英文) [显示名]
@addtogroup 作为指定名的模块的成员            模块名(英文) [显示名]
@name       按用途分,以便理解全局变量/宏的用途 显示名(中文)           @{ 变量/宏 @}
```

#### 函数信息

```
@brief     摘要
@param     参数说明
@param[in]      输入参数
@param[out]     输出参数
@param[in,out]  输入输出参数
@return    返回值说明
@retval    特定返回值说明 [eg:@retval NULL 空字符串][@retval !NULL 非空字符串]
@exception 可能产生的异常描述
@enum      引用了某个枚举,Doxygen会在引用处生成链接
@var       引用了某个变量
@class     引用某个类 [eg: @class CTest "inc/class.h"]
@see       参考链接,函数重载的情况下,要带上参数列表以及返回值
@todo      todo注解
@pre       前置条件说明
@par       [段落标题] 开创新段落,一般与示例代码联用 
@code      示例代码开始 e.g. [code{.cpp}]
@ endcode  示例代码结束
```

#### 提醒信息

```
@note      注解
@attention 注意
@warning   警告
@bug       问题
@def       宏定义说明
```
 

#### 生成

文档目录下执行
doxygen Doxyfile

### 函数

应尽可能多的使用模板函数

构造函数只进行没有实际意义的初始化,通过Init(),Setup()等函数进行具体构造.








