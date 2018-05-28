/* Copyright 2018 Gerald Baier
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

#include "conversion.h"
#include "logging.h"

cl::Buffer
nlsar::data_to_covmat(const insar_data& sub_insar_data,
                      const cl::Context& context,
                      const cl::CommandQueue& cmd_queue,
                      covmat_create& cl_routine,
                      const buffer_sizes& buf_sizes)
{
  // overlapped dimension, large enough to include the complete padded data to
  // compute the similarities; also includes overlap for spatial averaging
  const int height_overlap_avg = sub_insar_data.height();
  const int width_overlap_avg  = sub_insar_data.width();

  cl::Buffer device_ampl_master{context,
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                buf_sizes.io_data(),
                                sub_insar_data.ampl_master(),
                                NULL};
  cl::Buffer device_ampl_slave{context,
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               buf_sizes.io_data(),
                               sub_insar_data.ampl_slave(),
                               NULL};
  cl::Buffer device_phase{context,
                          CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                          buf_sizes.io_data(),
                          sub_insar_data.phase(),
                          NULL};

  cl::Buffer covmat{
      context, CL_MEM_READ_WRITE, buf_sizes.io_covmat(), NULL, NULL};

  cl_routine.timed_run(cmd_queue,
                 device_ampl_master,
                 device_ampl_slave,
                 device_phase,
                 covmat,
                 height_overlap_avg,
                 width_overlap_avg);

  return covmat;
}

void
nlsar::covmat_to_data(const cl::Buffer& covmat_filt,
                      insar_data& sub_insar_data,
                      const cl::Context& context,
                      const cl::CommandQueue& cmd_queue,
                      covmat_decompose& cl_routine,
                      const buffer_sizes& buf_sizes)
{
  const int height_overlap_avg = sub_insar_data.height();
  const int width_overlap_avg  = sub_insar_data.width();

  cl::Buffer device_ref_filt{
      context, CL_MEM_READ_WRITE, buf_sizes.io_data(), NULL, NULL};
  cl::Buffer device_phase_filt{
      context, CL_MEM_READ_WRITE, buf_sizes.io_data(), NULL, NULL};
  cl::Buffer device_coh_filt{
      context, CL_MEM_READ_WRITE, buf_sizes.io_data(), NULL, NULL};

  LOG(DEBUG) << "covmat_decompose";
  cl_routine.run(cmd_queue,
                 covmat_filt,
                 device_ref_filt,
                 device_phase_filt,
                 device_coh_filt,
                 height_overlap_avg,
                 width_overlap_avg);

  //***************************************************************************
  //
  // copying back result and clean up
  //
  //***************************************************************************
  LOG(DEBUG) << "copying sub result";
  cmd_queue.enqueueReadBuffer(device_ref_filt,
                              CL_TRUE,
                              0,
                              buf_sizes.io_data(),
                              sub_insar_data.ref_filt(),
                              NULL,
                              NULL);
  cmd_queue.enqueueReadBuffer(device_phase_filt,
                              CL_TRUE,
                              0,
                              buf_sizes.io_data(),
                              sub_insar_data.phase_filt(),
                              NULL,
                              NULL);
  cmd_queue.enqueueReadBuffer(device_coh_filt,
                              CL_TRUE,
                              0,
                              buf_sizes.io_data(),
                              sub_insar_data.coh_filt(),
                              NULL,
                              NULL);
}
