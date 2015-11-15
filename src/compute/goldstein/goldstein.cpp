#include "despeckcl.h"

#include <CL/cl.h>
#include <chrono>
#include <string.h> // for memcpy
#include <vector>
#include <string>
#include <omp.h>

#include "cl_wrappers.h"
#include "insar_data.h"
#include "tile_iterator.h"
#include "goldstein_filter_sub_image.h"
#include "sub_images.h"
#include "clcfg.h"
#include "logging.h"
#include "timings.h"


int despeckcl::goldstein(float* ampl_master,
                         float* ampl_slave,
                         float* dphase,
                         float* ampl_filt,
                         float* dphase_filt,
                         float* coh_filt,
                         const int height,
                         const int width,
                         const int patch_size,
                         const int overlap,
                         const float alpha,
                         std::vector<std::string> enabled_log_levels)
{
    timings::map tm;

    logging_setup(enabled_log_levels);

    LOG(INFO) << "filter parameters";
    LOG(INFO) << "patch_size: " << patch_size;
    LOG(INFO) << "alpha: "      << alpha;
    LOG(INFO) << "overlap: "    << overlap;

    LOG(INFO) << "data dimensions";
    LOG(INFO) << "height: " << height;
    LOG(INFO) << "width: "  << width;

    // legacy opencl setup
    cl::Context context = opencl_setup();

    const int sub_image_size = 256;

    // new build kernel interface
    std::chrono::time_point<std::chrono::system_clock> start, end;
    std::chrono::duration<double> elapsed_seconds = end-start;
    start = std::chrono::system_clock::now();
    VLOG(0) << "Building kernels";
    goldstein::cl_wrappers goldstein_cl_wrappers (context, 16);
    end = std::chrono::system_clock::now();
    elapsed_seconds = end-start;
    VLOG(0) << "Time it took to build all kernels: " << elapsed_seconds.count() << "secs";

    // prepare data
    insar_data total_image{ampl_master, ampl_slave, dphase,
                           ampl_filt, dphase_filt, coh_filt,
                           height, width};

    // filtering
    start = std::chrono::system_clock::now();
    LOG(INFO) << "starting filtering";
#pragma omp parallel shared(total_image)
{
#pragma omp master
    {
        for( auto imgtile : tile_iterator(total_image, sub_image_size, overlap, overlap) ) {
#pragma omp task firstprivate(imgtile)
        {
        try {
            timings::map tm_sub = filter_sub_image(context,
                                                   goldstein_cl_wrappers, // opencl stuff
                                                   imgtile.get(), // data
                                                   patch_size,
                                                   overlap,
                                                   alpha);
#pragma omp critical
            tm = timings::join(tm, tm_sub);
        } catch (cl::Error error) {
            LOG(ERROR) << error.what() << "(" << error.err() << ")";
            LOG(ERROR) << "ERR while filtering sub image";
            std::terminate();
        }
        imgtile.write(total_image);
        }
    }
#pragma omp taskwait
    }
}
    LOG(INFO) << "filtering done";
    timings::print(tm);
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> duration = end-start;
    std::cout << "filtering ran for " << duration.count() << " secs" << std::endl;

    memcpy(ampl_filt,   total_image.amp_filt, sizeof(float)*height*width);
    memcpy(dphase_filt, total_image.phi_filt, sizeof(float)*height*width);
    memcpy(coh_filt,    total_image.coh_filt, sizeof(float)*height*width);

    return 0;
}
