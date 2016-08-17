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

#ifndef COMPUTE_H
#define COMPUTE_H

#include <string>
#include <chrono>
#include <iostream>

template<typename Derived>
class routine_env
{
    public:
        std::string routine_name;
        std::chrono::duration<double> elapsed_seconds{};

        template<typename... Args>
        double timed_run(Args... args)
        {
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();

            static_cast<Derived*>(this)->run(args...);

            end = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end-start;
            this->elapsed_seconds += duration;
            return duration.count();
        }

};

#endif
