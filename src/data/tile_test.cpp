#include <stdexcept>

#include "tile.h"

#include "gtest/gtest.h"

TEST(SLICE, exceptions) {
    ASSERT_NO_THROW(slice(2, 4));
    ASSERT_THROW(slice(4, 2), std::range_error);

    slice s1{2, 6};
    ASSERT_NO_THROW(s1.get_sub(1, -1));
    ASSERT_THROW(s1.get_sub(1, 1), std::range_error);
    ASSERT_THROW(s1.get_sub(-1, -1), std::range_error);
    ASSERT_THROW(s1.get_sub(4, -4), std::range_error);
}

TEST(SLICE, concatenation) {
    slice s1{0, 10};
    slice s2 = s1.get_sub(2, -2);

    ASSERT_EQ(s2.start, 2);
    ASSERT_EQ(s2.stop, 8);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
