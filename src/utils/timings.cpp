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
