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
        void timed_run(Args... args)
        {
            std::chrono::time_point<std::chrono::system_clock> start, end;
            start = std::chrono::system_clock::now();

            static_cast<Derived*>(this)->run(args...);

            end = std::chrono::system_clock::now();
            elapsed_seconds += end-start;
        }

        void print_elapsed_time(void)
        {
            std::cout << "elapsed time for " << routine_name << ": " << elapsed_seconds.count() << "s\n";
        }
};

#endif
