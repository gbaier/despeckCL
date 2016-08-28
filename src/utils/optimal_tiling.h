#ifndef OPTIMAL_TILING_H
#define OPTIMAL_TILING_H

#include <array>
#include <vector>
#include <utility>
#include <math.h>

template<int N>
constexpr std::array<int, N> range(int step) {
  std::array<int, N> range_val{0};
  for(int i=0; i<N; i++) {
    range_val[i] = (i+1)*step;
  }
  return range_val;
}

std::vector<std::pair<int, int>> all_pairs(std::vector<int> range);

class get_nm_tiles {
  private:
    const int _img_height;
    const int _img_width;
    const int _overlap;

  public:
    get_nm_tiles(int img_height, int img_width, int overlap)
        : _img_height(img_height), _img_width(img_width), _overlap(overlap) {};

    std::pair<int, int> operator()(std::pair<int, int> tile_dim) {
      const int n_height = std::ceil((_img_height+2*_overlap)/static_cast<float>(tile_dim.first-2*_overlap));
      const int n_width = std::ceil((_img_width+2*_overlap)/static_cast<float>(tile_dim.second-2*_overlap));
      return std::make_pair(n_height, n_width);
    }
};

size_t tiled_img_npixels(std::pair<int, int> tile_dim, std::pair<int, int> nm_tiles, int overlap);

std::vector<std::pair<int, int>> retain_small_offcut_tiles(
    std::vector<std::pair<int, int>> tiles,
    size_t img_height,
    size_t img_width,
    int overlap,
    float offcut = 1.1f);

std::pair<int, int> biggest_tile(std::vector<std::pair<int, int>> tiles);

float scale_factor(std::pair<int, int> p);
#endif
