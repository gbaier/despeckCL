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

#include "clcfg.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "../utils/easylogging++.h"

std::vector<cl::Device> get_platform_devs(int platform_id) {
    LOG(INFO) << "OpenCL setup";

    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (cl::Error &err) {
        LOG(FATAL) << err.what() << "(" << err.err() << ")";
    }

    LOG(INFO) << "found the following platforms";
    for(auto pf : platforms) {
        std::string platform_vendor;
        pf.getInfo(CL_PLATFORM_VENDOR, &platform_vendor);
        LOG(INFO) << "platform vendor: " << platform_vendor;

        std::string platform_name;
        pf.getInfo(CL_PLATFORM_NAME, &platform_name);
        LOG(INFO) << "platform name: " << platform_name;
    }
    LOG(INFO) << "selecting platform: " << platform_id << "\n\n";

    cl::Platform selected_platform{platforms[platform_id]};

    std::vector<cl::Device> devices;
    selected_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    
    LOG(INFO) << "found the following devices";
    for(const auto& dev : devices) {
        LOG(INFO) << "device name: " << dev.getInfo<CL_DEVICE_NAME>();
        LOG(INFO) << "device vendor id: " << dev.getInfo<CL_DEVICE_VENDOR_ID>();

        std::vector<size_t> max_work_item_sizes(3);
        dev.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_work_item_sizes);
        LOG(INFO) << "max work item sizes: " << max_work_item_sizes[0] << ", "
                                             << max_work_item_sizes[1] << ", "
                                             << max_work_item_sizes[2];

        LOG(INFO) << "global memory size: " << dev.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
        LOG(INFO) << "maximum memory allocation size: " << dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

        LOG(INFO) << "device profile: " << dev.getInfo<CL_DEVICE_PROFILE>();
        LOG(INFO) << "device version: " << dev.getInfo<CL_DEVICE_VERSION>();
        LOG(INFO) << "driver version: " << dev.getInfo<CL_DRIVER_VERSION>();
        LOG(INFO) << "device opencl_c version" << dev.getInfo<CL_DEVICE_OPENCL_C_VERSION>();
        LOG(INFO) << "device extensions: " << dev.getInfo<CL_DEVICE_EXTENSIONS>();
    }

    return devices;
}
