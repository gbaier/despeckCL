#include "clcfg.h"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>

#include "../utils/easylogging++.h"

cl::Context opencl_setup(void)
{
    LOG(DEBUG) << "OpenCL setup";

    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (cl::Error err) {
        LOG(FATAL) << err.what() << "(" << err.err() << ")";
    }

    LOG(DEBUG) << "found the following platforms";
    for(auto pf : platforms) {
        std::string platform_vendor;
        pf.getInfo(CL_PLATFORM_VENDOR, &platform_vendor);
        LOG(DEBUG) << "platform vendor: " << platform_vendor;

        std::string platform_name;
        pf.getInfo(CL_PLATFORM_NAME, &platform_name);
        LOG(DEBUG) << "platform name: " << platform_name;
    }
    LOG(DEBUG) << "selecting first platform";

    cl::Platform selected_platform = platforms[0];


    std::vector<cl::Device> devices;
    selected_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    
    LOG(DEBUG) << "found the following devices";
    for(auto dev : devices) {
        std::string device_name;
        dev.getInfo(CL_DEVICE_NAME, &device_name);
        LOG(DEBUG) << "device name: " << device_name;

        std::vector<size_t> max_work_item_sizes(3);
        dev.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_work_item_sizes);
        LOG(DEBUG) << "max work item sizes: " << max_work_item_sizes[0] << ", "
                                              << max_work_item_sizes[1] << ", "
                                              << max_work_item_sizes[2];

        std::string device_profile;
        dev.getInfo(CL_DEVICE_PROFILE, &device_profile);
        LOG(DEBUG) << "device profile: " << device_profile;

        std::string device_version;
        dev.getInfo(CL_DEVICE_VERSION, &device_version);
        LOG(DEBUG) << "device version: " << device_version;

        std::string driver_version;
        dev.getInfo(CL_DRIVER_VERSION, &driver_version);
        LOG(DEBUG) << "driver version: " << driver_version;

        std::string device_opencl_c_version;
        dev.getInfo(CL_DEVICE_OPENCL_C_VERSION, &device_opencl_c_version);
        LOG(DEBUG) << "device opencl_c_version: " << device_opencl_c_version;

        std::string device_extensions;
        dev.getInfo(CL_DEVICE_EXTENSIONS, &device_extensions);
        LOG(DEBUG) << "device extensions: " << device_extensions << std::endl;
    }

    cl::Context context (devices);
                         
    return context;
}
