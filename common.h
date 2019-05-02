/*
@brief common library only include header.
@author guobao.v@gmail.com
*/
#ifdef _MSC_VER
#pragma once
#endif

#ifndef _COMMON_H_
#define _COMMON_H_

#include <common/cmdline.h>
#include <iostream>

namespace common {
	const std::string _TAG = "common";

#define free_s(p) if(p!=nullptr){free(p);p=nullptr;}
#define delete_s(p) if(p!=nullptr){delete(p);p=nullptr;}
#define deleteA_s(p) if(p!=nullptr){delete[](p);p=nullptr;}
#define Release_s(p) if(p!=nullptr){p->Release();p=nullptr;}
#define release_s(p) if(p!=nullptr){p->release();p=nullptr;}
#define zeroset(x) (memset((&x),0,sizeof(x)))

}
#endif // _COMMON_H_