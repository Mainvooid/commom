/*
@brief unit test for precomm.hpp & common.hpp
@author guobao.v@gmail.com
*/
#include "gtest/gtest.h"
#include <common/precomm.hpp>
#include <common.hpp>

using namespace common;
using std::cout;
using std::endl;

size_t fibonacci_1(size_t n) {
    return (n < 2) ? n : fibonacci_1(n - 1) + fibonacci_1(n - 2);
}
size_t fibonacci_2(size_t n) {
    return (n < 2) ? n : cache_fn(fibonacci_2)(n - 1) + cache_fn(fibonacci_2)(n - 2);
}

TEST(common, cache_fn__no_cache) {
    EXPECT_EQ(fibonacci_1(35), 9227465);
}
TEST(common, cache_fn__cache) {
    EXPECT_EQ(fibonacci_2(35), 9227465);
}
TEST(common, cache_fn__speed) {
    auto t1 = getFnDuration(fibonacci_1)(35);//44ms (为45时,为5000ms)
    auto t2 = getFnDuration(fibonacci_2)(35);//1ms  (为1000时,为2ms)
    EXPECT_GE(t1, t2);
}