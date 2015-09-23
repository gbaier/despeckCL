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
