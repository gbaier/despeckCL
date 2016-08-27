#include "gtest/gtest.h"

#include "logging.h"

int main(int argc, char **argv)
{
    logging_setup({});
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
