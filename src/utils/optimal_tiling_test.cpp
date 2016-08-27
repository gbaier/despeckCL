#include "optimal_tiling.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <algorithm>

TEST(OPTIMAL_TILING, range) {
  std::array<int, 2> arr {4, 8};
  ASSERT_THAT(range<2>(4), arr);
}

TEST(OPTIMAL_TILING, combinations) {
  std::vector<int> in {4, 8, 12, 16};
  std::vector<std::pair<int, int>> out{ {4, 4},  {4, 8},  {4, 12},  {4, 16},
                                        {8, 4},  {8, 8},  {8, 12},  {8, 16},
                                        {12, 4}, {12, 8}, {12, 12}, {12, 16},
                                        {16, 4}, {16, 8}, {16, 12}, {16, 16} };
  ASSERT_THAT(all_pairs(in), out);
}

TEST(OPTIMAL_TILING, get_nm_tiles) {
  const int img_height = 11;
  const int img_width = 17;
  get_nm_tiles gt{img_height, img_width};
  std::vector<std::pair<int, int>> in  {{4, 4}, {4, 8}};
  std::vector<std::pair<int, int>> out {{3, 5}, {3, 3}};
  std::transform(std::begin(in), std::end(in), std::begin(in), gt);
  ASSERT_THAT(in, out);
}

TEST(OPTIMAL_TILING, tiled_img_npixels) {
  std::pair<int, int> tiles_in_img {4, 5};
  std::pair<int, int> tile_size {10, 10};
  ASSERT_THAT(tiled_img_npixels(tile_size, tiles_in_img), 2000);
}

TEST(OPTIMAL_TILING, retain_small_offcut_tiles) {
  const int img_height = 12;
  const int img_width = 16;
  std::vector<std::pair<int, int>> in  {{4, 4}, {4, 8}, {8, 4}, {8, 8}};
  std::vector<std::pair<int, int>> out {{4, 4}, {4, 8}};
  ASSERT_THAT(retain_small_offcut_tiles(in, img_height, img_width), out);
}


TEST(OPTIMAL_TILING, biggest_tile) {
  std::vector<std::pair<int, int>> in  {{4, 4}, {4, 8}, {8, 4}, {8, 8}};
  std::pair<int, int> out {8, 8};
  ASSERT_THAT(biggest_tile(in), out);
}

