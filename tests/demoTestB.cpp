#include "gtest/gtest.h"

TEST(DemoTestB, DemoTestb1) {
    std::cout << "b1: " << getpid() << std::endl;
    EXPECT_TRUE(true);
}

TEST(DemoTestB, DemoTestb2) {
    std::cout << "b2: " << getpid() << std::endl;
    EXPECT_TRUE(true);
}