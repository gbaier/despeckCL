#ifndef MAP_FILTER_TILES_H
#define MAP_FILTER_TILES_H

#include "data.h"
#include "logging.h"
#include "tile_iterator.h"
#include "tile.h"

template <typename Type, typename Filter, typename Clroutines, typename... Params>
timings::map
map_filter_tiles(Filter func,
                 Type& total_image_in,
                 Type& total_image_out,
                 cl::Context context,
                 const Clroutines& cl_wrappers,
                 std::pair<int, int> tile_dims,
                 int overlap,
                 Params... parameters)
{
  timings::map tm;
  LOG(INFO) << "starting filtering";
#pragma omp parallel shared(total_image_in, total_image_out)
  {
#pragma omp master
    {
      for (auto t : tile_iterator(total_image_in.height(),
                                  total_image_in.width(),
                                  tile_dims.first,
                                  tile_dims.second,
                                  overlap,
                                  overlap)) {
#pragma omp task firstprivate(t)
        {
          Type imgtile(tileget(total_image_in, t));
          try {
            timings::map tm_sub =
                func(context, cl_wrappers, imgtile, parameters...);
#pragma omp critical
            tm = timings::join(tm, tm_sub);
          } catch (cl::Error &error) {
            LOG(ERROR) << error.what() << "(" << error.err() << ")";
            LOG(ERROR) << "ERR while filtering sub image";
            std::terminate();
          }
          tile<2> rel_sub{t[0].get_sub(overlap, -overlap),
                          t[1].get_sub(overlap, -overlap)};
          tile<2> tsub{slice{overlap, imgtile.height() - overlap},
                       slice{overlap, imgtile.width() - overlap}};
          Type imgtile_sub(tileget(imgtile, tsub));
          tilecpy(total_image_out, imgtile_sub, rel_sub);
        }
      }
#pragma omp taskwait
    }
  }
  LOG(INFO) << "filtering done";
  return tm;
}

#endif
