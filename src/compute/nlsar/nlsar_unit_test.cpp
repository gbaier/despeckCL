#include "gtest/gtest.h"

#include "easylogging++.h"
INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
