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

template<typename Type>
class sar_data {
    public:
        std::unique_ptr<Type[]> _data;
        int height;
        int width;
        int size;

        // takes ownership
        sar_data(std::unique_ptr<Type[]> data, int height, int width, int size)
            : _data(std::move(data)), height(height), width(width), size(size)
        {
        }

        sar_data(const sar_data& other) = delete;

        sar_data(sar_data&& other) noexcept
        {
          std::swap(height, other.height);
          std::swap(width, other.width);
          std::swap(size, other.size);
          std::swap(_data, other._data);
        }

        sar_data&
        operator=(sar_data&& other) noexcept
        {
          std::swap(height, other.height);
          std::swap(width, other.width);
          std::swap(size, other.size);
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
  sar_data<float> _cont;

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

  // takes ownership
  insar_data(std::unique_ptr<float[]> data, int height, int width)
      : _cont(std::move(data), height, width, 6)
  {
  }

  // pubic interface that abstracts the internel data representation
  int height() const { return _cont.height; };
  int width() const  { return _cont.width; };
  int size()   const  { return _cont.size; };
  int dim()   const  { return 2; };
  float * data()  const  { return _cont.data(); };

  float* ampl_master() const { return _cont._data.get(); };
  float* ampl_slave() const  { return _cont._data.get() +     height() * width(); };
  float* phase() const       { return _cont._data.get() + 2 * height() * width(); };
  
  float* ref_filt() const    { return _cont._data.get() + 3 * height() * width(); };
  float* phase_filt() const  { return _cont._data.get() + 4 * height() * width(); };
  float* coh_filt() const    { return _cont._data.get() + 5 * height() * width(); };
};


class covmat_data
{
 private:
  // storage for raw and filtered covariance matrix: first factor of 2
  // and also real and imaginary part: second factor of 2
  sar_data<float> _cont;
  const int _dim;

 public:
  // Allocates memory and copies data.
  // This is for interfacing with C-libraries/programs
  // or Python via SWIG.
  covmat_data(float* covmat_raw,
              float* covmat_filt,
              int height,
              int width,
              int dim);

  // takes ownership
  covmat_data(std::unique_ptr<float[]> data, int height, int width, int dim)
      : _cont(std::move(data), height, width, 2*2*dim*dim), _dim(dim)
  {
  }

  // pubic interface that abstracts the internel data representation
  int height() const { return _cont.height; };
  int width() const  { return _cont.width; };
  int size()   const  { return _cont.size; };
  int dim()   const  { return _dim; };
  float * data()  const  { return _cont.data(); };

  float* covmat_raw()  const { return _cont._data.get(); };
  float* covmat_filt() const { return _cont._data.get() + 2 * dim() * dim() * height() * width(); };
};

class ampl_data
{
 private:
  sar_data<float> _cont;

 public:
  // Allocates memory and copies data.
  // This is for interfacing with C-libraries/programs
  // or Python via SWIG.
  ampl_data(float* ampl, float* ref_filt, int height, int width);

  // takes ownership
  ampl_data(std::unique_ptr<float[]> data, int height, int width)
      : _cont(std::move(data), height, width, 2)
  {
  }

  // pubic interface that abstracts the internel data representation
  int height() const { return _cont.height; };
  int width() const  { return _cont.width; };
  int size()   const  { return _cont.size; };
  float * data()  const  { return _cont.data(); };

  float* ampl()     const { return _cont._data.get(); };
  float* ref_filt() const { return _cont._data.get() + height() * width(); };
};

template<typename DataType>
DataType tileget(const DataType& img_data, tile<2> sub) {
  auto data_sub = get_sub_image(img_data.data(),
                                img_data.height(),
                                img_data.width(),
                                img_data.size(),
                                sub[0].start,
                                sub[1].start,
                                sub[0].stop - sub[0].start,
                                sub[1].stop - sub[1].start);
  return DataType{std::move(data_sub),
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start};
}


covmat_data tileget(const covmat_data& img_data, tile<2> sub);

// copy img_tile to img_data defined by sub
// akin to memcpy
template<typename DataType>
void tilecpy(DataType& img_data, const DataType& img_tile, tile<2> sub) {
  write_sub_image(img_data.data(),
                  img_data.height(),
                  img_data.width(),
                  img_data.size(),
                  img_tile.data(),
                  sub[0].start,
                  sub[1].start,
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start,
                  0);
}

#endif
