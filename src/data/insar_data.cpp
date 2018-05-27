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

#include "insar_data.h"
#include "sub_images.h"

#include <algorithm>

insar_data::insar_data(std::unique_ptr<float[]> ampl_master,
                       std::unique_ptr<float[]> ampl_slave,
                       std::unique_ptr<float[]> phase,
                       std::unique_ptr<float[]> ref_filt,
                       std::unique_ptr<float[]> phase_filt,
                       std::unique_ptr<float[]> coh_filt,
                       int height,
                       int width) 
            : _ampl_master(std::move(ampl_master)),
              _ampl_slave(std::move(ampl_slave)),
              _phase(std::move(phase)),
              _ref_filt(std::move(ref_filt)),
              _phase_filt(std::move(phase_filt)),
              _coh_filt(std::move(coh_filt)),
              height(height),
              width(width)
        {
        }

insar_data::insar_data(float * a1,
                       float * a2,
                       float * dp,
                       float * ref_filt,
                       float * phi_filt,
                       float * coh_filt,
                       int height,
                       int width) : height(height), width(width)
{
    const size_t size = height*width;

    this->_ampl_master = std::make_unique<float[]>(size);
    this->_ampl_slave  = std::make_unique<float[]>(size);
    this->_phase       = std::make_unique<float[]>(size);
    this->_ref_filt    = std::make_unique<float[]>(size);
    this->_phase_filt  = std::make_unique<float[]>(size);
    this->_coh_filt    = std::make_unique<float[]>(size);

    std::copy(a1, a1+size, this->_ampl_master.get());
    std::copy(a2, a2+size, this->_ampl_slave.get());
    std::copy(dp, dp+size, this->_phase.get());
    std::copy(ref_filt, ref_filt+size, this->_ref_filt.get());
    std::copy(phi_filt, phi_filt+size, this->_phase_filt.get());
    std::copy(coh_filt, coh_filt+size, this->_coh_filt.get());
}

insar_data::insar_data(insar_data &&other) noexcept
{
    std::swap(height, other.height);
    std::swap(width, other.width);
    std::swap(_ampl_master, other._ampl_master);
    std::swap(_ampl_slave, other._ampl_slave);
    std::swap(_phase, other._phase);
    std::swap(_ref_filt, other._ref_filt);
    std::swap(_phase_filt, other._phase_filt);
    std::swap(_coh_filt, other._coh_filt);
}

insar_data& insar_data::operator=(insar_data &&other) noexcept {
    std::swap(height, other.height);
    std::swap(width, other.width);
    std::swap(_ampl_master, other._ampl_master);
    std::swap(_ampl_slave, other._ampl_slave);
    std::swap(_phase, other._phase);
    std::swap(_ref_filt, other._ref_filt);
    std::swap(_phase_filt, other._phase_filt);
    std::swap(_coh_filt, other._coh_filt);
    return *this;
}

insar_data tileget(const insar_data& img_data, tile<2> sub) {
  auto _ampl_master_sub = get_sub_image(img_data.ampl_master(),
                                        img_data.height,
                                        img_data.width,
                                        sub[0].start,
                                        sub[1].start,
                                        sub[0].stop - sub[0].start,
                                        sub[1].stop - sub[1].start);
  auto _ampl_slave_sub  = get_sub_image(img_data.ampl_slave(),
                                       img_data.height,
                                       img_data.width,
                                       sub[0].start,
                                       sub[1].start,
                                       sub[0].stop - sub[0].start,
                                       sub[1].stop - sub[1].start);
  auto _phase_sub       = get_sub_image(img_data.phase(),
                                  img_data.height,
                                  img_data.width,
                                  sub[0].start,
                                  sub[1].start,
                                  sub[0].stop - sub[0].start,
                                  sub[1].stop - sub[1].start);
  auto ref_filt_sub     = get_sub_image(img_data.ref_filt(),
                                    img_data.height,
                                    img_data.width,
                                    sub[0].start,
                                    sub[1].start,
                                    sub[0].stop - sub[0].start,
                                    sub[1].stop - sub[1].start);
  auto phi_filt_sub     = get_sub_image(img_data.phase_filt(),
                                    img_data.height,
                                    img_data.width,
                                    sub[0].start,
                                    sub[1].start,
                                    sub[0].stop - sub[0].start,
                                    sub[1].stop - sub[1].start);
  auto coh_filt_sub     = get_sub_image(img_data.coh_filt(),
                                    img_data.height,
                                    img_data.width,
                                    sub[0].start,
                                    sub[1].start,
                                    sub[0].stop - sub[0].start,
                                    sub[1].stop - sub[1].start);
  return insar_data{std::move(_ampl_master_sub),
                    std::move(_ampl_slave_sub),
                    std::move(_phase_sub),
                    std::move(ref_filt_sub),
                    std::move(phi_filt_sub),
                    std::move(coh_filt_sub),
                    sub[0].stop - sub[0].start,
                    sub[1].stop - sub[1].start};
}

// copy img_tile to img_data defined by sub
// akin to memcpy
void tilecpy(insar_data& img_data, const insar_data& img_tile, tile<2> sub) {
  write_sub_image(img_data.ref_filt(),
                  img_data.height,
                  img_data.width,
                  img_tile.ref_filt(),
                  sub[0].start,
                  sub[1].start,
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start,
                  0);
  write_sub_image(img_data.phase_filt(),
                  img_data.height,
                  img_data.width,
                  img_tile.phase_filt(),
                  sub[0].start,
                  sub[1].start,
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start,
                  0);
  write_sub_image(img_data.coh_filt(),
                  img_data.height,
                  img_data.width,
                  img_tile.coh_filt(),
                  sub[0].start,
                  sub[1].start,
                  sub[0].stop - sub[0].start,
                  sub[1].stop - sub[1].start,
                  0);
}
