/*
@brief tests entry point
@author guobao.v@gmail.com
*/
#include "gtest/gtest.h"

#if defined(_DEBUG) || defined(DEBUG)
#pragma comment(lib,"gtestd.lib")
#else
#pragma comment(lib,"gtest.lib")
#endif

int main(int argc, char** argv)
{
    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    system("pause");
    return result;
}