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

#ifndef CLCFG_H
#define CLCFG_H

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#include <string>
#include <iostream>

#include "../compute/compute_env.h"
#include "../utils/easylogging++.h"

cl::Context opencl_setup(void);

std::string read_cl_file(std::string kernel_path);

template<typename Derived>
class kernel_env : public routine_env<Derived>
{
    public:
        kernel_env(size_t block_size,
                   cl::Context context) : block_size(block_size), context(context) {}

        kernel_env(const kernel_env& other) : block_size(other.block_size), context(other.context) {}


        size_t block_size;
        cl::Context context;

        static constexpr const char* kernel_source {"SOURCE"};
        cl::Program program;

   protected:
        std::string default_build_opts(void)
        {
            return std::string{"-Werror -cl-std=CL1.2"};
        }

        std::string build_opts(void)
        {
            return default_build_opts();
        }

        cl::Program build_program(std::string build_opts, std::string kernel_source)
        {
            std::vector<cl::Device> devices;
            static_cast<Derived*>(this)->context.getInfo(CL_CONTEXT_DEVICES, &devices);

            std::string routine_name  = static_cast<Derived*>(this)->routine_name;
            LOG(DEBUG) << "Building program for: " << routine_name;

            cl::Program program{static_cast<Derived*>(this)->context, kernel_source};
            try {
                program.build(devices, build_opts.c_str());
            } catch (cl::Error &error) {
                LOG(ERROR) << "ERROR";
                LOG(ERROR) << error.what() << "(" << error.err() << ")";
                std::string build_log;
                program.getBuildInfo(devices[0], CL_PROGRAM_BUILD_LOG, &build_log);
                LOG(ERROR) << build_log;
                std::terminate();
            }
            return program;
        }

        cl::Kernel build_kernel(cl::Program program, std::string routine_name)
        {
            LOG(DEBUG) << "Building kernel for: " << routine_name;
            try {
                return cl::Kernel{program, routine_name.c_str()};
            } catch (cl::Error &error) {
                LOG(ERROR) << error.what() << "(" << error.err() << ")";
                std::terminate();
            }
        }

   public:
        template<typename... Args>
        double timed_run(cl::CommandQueue cmd_queue, Args... args)
        {
            std::string routine_name = static_cast<Derived*>(this)->routine_name;
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();

            try {
                static_cast<Derived*>(this)->run(cmd_queue, args...);
            } catch (cl::Error &error) {
                LOG(ERROR) << "ERR while running kernel: " << routine_name;
                LOG(ERROR) << error.what() << "(" << error.err() << ")";
                std::terminate();
            }
#ifdef PERF
            cmd_queue.finish();
#endif

            end = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end-start;
            this->elapsed_seconds += duration;
            return duration.count();
        }
};

template<typename Derived>
class kernel_env_single : public kernel_env<Derived>
{
    using kernel_env<Derived>::kernel_env;

    public:
        cl::Kernel kernel;
};

template<typename Derived>
class kernel_env_build : public kernel_env<Derived>
{
    public:
        cl::Kernel kernel;

        kernel_env_build(size_t block_size,
                         cl::Context context) : kernel_env<Derived>(block_size, context)
        {
            std::string routine_name {static_cast<Derived*>(this)->routine_name};
            this->program = kernel_env<Derived>::build_program(kernel_env<Derived>::build_opts(), static_cast<Derived*>(this)->kernel_source);
            this->kernel  = kernel_env<Derived>::build_kernel(this->program, routine_name);
        }

        kernel_env_build(const kernel_env_build& other) : kernel_env<Derived>(other)
        {
            std::string routine_name {static_cast<Derived*>(this)->routine_name};
            this->program = other.program;
            this->kernel  = kernel_env<Derived>::build_kernel(this->program, routine_name);
        }

};
#endif
