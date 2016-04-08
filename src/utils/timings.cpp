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

#include "timings.h"

#include <algorithm>
#include <iomanip>

#include "easylogging++.h"

timings::map timings::join(const map& tm1, const map& tm2)
{
    map new_tm = tm1;

    new_tm = std::accumulate( tm2.begin(), tm2.end(), new_tm,
            []( auto& tm, const auto& p )
            {
            return (tm[p.first] += p.second, tm);
            } );

    return new_tm;
}

double timings::total_time(const map &tm)
{
    return std::accumulate( tm.begin(), tm.end(), 0.0,
            []( const double total, const auto& p )
            {
            return total + p.second;
            } );
}

void timings::print(const map& tm)
{
    std::vector<std::pair<std::string, double>> timings;
    for(auto tel : tm) {
        timings.push_back(tel);
    }
    std::sort(timings.begin(), timings.end(), [] (auto el1, auto el2) { return el1.second < el2.second; } );
    for(auto p : timings) {
        LOG(INFO) << "kernel execution time for " << std::left << std::setw(30) << p.first \
                  << " is " << std::setw(10) << std::fixed << std::setprecision(5) << p.second << " secs";
    }
    LOG(INFO) << "total kernel execution time is " << total_time(tm) << " secs";
}
