/* Copyright 2015, 2016 Gerald Baier
 *
 * This file is part of despeckCL.
 *
 * despeckCL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * despeckCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with despeckCL. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef INSAR_DATA_H
#define INSAR_DATA_H

#include <algorithm>
#include <memory>

#include "sub_images.h"
#include "tile.h"

template<typename Type, size_t D>
class sar_data {
    public:
        std::unique_ptr<Type[]> _data;
        int height;
        int width;
        const int dim = D;

        // takes ownership
        sar_data(std::unique_ptr<Type[]> data, int height, int width)
            : _data(std::move(data)), height(height), width(width)
        {
        }

        sar_data(const sar_data& other) = delete;

        sar_data(sar_data&& other) noexcept
        {
          std::swap(height, other.height);
          std::swap(width, other.width);
          std::swap(_data, other._data);
        }

        sar_data&
        operator=(sar_data&& other) noexcept
        {
          std::swap(height, other.height);
          std::swap(width, other.width);
          std::swap(_data, other._data);
          return *this;
        }

        ~sar_data() {};

        float * data()        const { return _data.get(); };
};

#include <memory>

class insar_data
{
 private:
  sar_data<float, 6> _cont;

 public:
  // Allocates memory and copies data.
  // This is for interfacing with C-libraries/programs
  // or Python via SWIG.
  insar_data(float* a1,
             float* a2,
             float* dp,
             float* ref_filt,
             float* phi_filt,
             float* coh_filt,
             int height,
             int width);

  explicit insar_data(sar_data<float, 6>&& cont) : _cont(std::move(cont)) {}
  // takes ownership
  insar_data(std::unique_ptr<float[]> data, int height, int width)
      : _cont(std::move(data), height, width)
  {
  }

  // pubic interface that abstracts the internel data representation
  int height() const { return _cont.height; };
  int width() const  { return _cont.width; };
  int dim()   const  { return _cont.dim; };
  float * data()  const  { return _cont.data(); };

  float* ampl_master() const { return _cont._data.get(); };
  float* ampl_slave() const  { return _cont._data.get() +     height() * width(); };
  float* phase() const       { return _cont._data.get() + 2 * height() * width(); };
  
  float* ref_filt() const    { return _cont._data.get() + 3 * height() * width(); };
  float* phase_filt() const  { return _cont._data.get() + 4 * height() * width(); };
  float* coh_filt() const    { return _cont._data.get() + 5 * height() * width(); };
};


class ampl_data
{
 private:
  sar_data<float, 1> _cont;

 public:
  // Allocates memory and copies data.
  // This is for interfacing with C-libraries/programs
  // or Python via SWIG.
  ampl_data(float* ampl, int height, int width);

  explicit ampl_data(sar_data<float, 1>&& cont) : _cont(std::move(cont)) {}
  // takes ownership
  ampl_data(std::unique_ptr<float[]> data, int height, int width)
      : _cont(std::move(data), height, width)
  {
  }

  // pubic interface that abstracts the internel data representation
  int height() const { return _cont.height; };
  int width() const  { return _cont.width; };
  int dim()   const  { return _cont.dim; };
  float * data()  const  { return _cont.data(); };

  float* ampl() const { return _cont._data.get(); };
};

template<typename DataType>
DataType tileget(const DataType& img_data, tile<2> sub) {
  auto data_sub = get_sub_image(img_data.data(),
                                img_data.height(),
                                img_data.width(),
                                img_data.dim(),
                                sub[0].start,
                                sub[1].start,
                                sub[0].stop - sub[0].start,
                                sub[1].stop - sub[1].start);
  return DataType{std::move(data_sub),
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start};
}

// copy img_tile to img_data defined by sub
// akin to memcpy
template<typename DataType>
void tilecpy(DataType& img_data, const DataType& img_tile, tile<2> sub) {
  write_sub_image(img_data.data(),
                  img_data.height(),
                  img_data.width(),
                  img_data.dim(),
                  img_tile.data(),
                  sub[0].start,
                  sub[1].start,
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start,
                  0);
}

#endif
