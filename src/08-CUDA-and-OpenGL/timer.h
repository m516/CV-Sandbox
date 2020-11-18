#pragma once
#include <chrono>
#include <utility>
#include <iostream>

class Timer
{
public:

    using clock_t = std::chrono::steady_clock;

private:

    clock_t::time_point start_time;

public:

    Timer():
        start_time(clock_t::now()) { }

    float getElapsedSeconds() const { 
        std::chrono::duration<float> diff = clock_t::now()-start_time;
        return diff.count(); 
    }

    void reset() { start_time = clock_t::now(); }
};