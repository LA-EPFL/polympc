#ifndef TIMER_HPP
#define TIMER_HPP

#include <stdio.h>

#include <cmath>
#include <chrono>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>

class Timer {
    /** OS dependent */
#ifdef __APPLE__
    using clock = std::chrono::system_clock;
#else
    using clock = std::chrono::high_resolution_clock;
#endif
    using time_point = std::chrono::time_point<clock>;

    time_point _start, _stop;
    std::vector<double> _samples;

public:
    time_point get_time()
    {
        return clock::now();
    }

    void tic()
    {
        _start = get_time();
    }

    void toc()
    {
        _stop = get_time();
        double t = std::chrono::duration<double, std::micro>(_stop - _start).count();
        _samples.push_back(t);
    }

    void clear()
    {
        _samples.clear();
    }

    const std::vector<double>& samples()
    {
        return _samples;
    }

    double sum()
    {
        return std::accumulate(_samples.begin(), _samples.end(), 0.0);
    }

    double mean()
    {
        if (_samples.size() == 0) {
            return 0;
        }
        return sum() / _samples.size();
    }

    std::tuple<double, double> mean_std()
    {
        double m, s;

        if (_samples.size() == 0) {
            return std::make_tuple(0.0, 0.0);
        }

        m = mean();

        std::vector<double> diff(_samples.size());
        std::transform(_samples.begin(), _samples.end(), diff.begin(),
                       [m](double x) {
            return x - m;
        }
                       );
        double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        s = sqrt(sq_sum / diff.size());

        return std::make_tuple(m, s);
    }

    void print()
    {
        double m, s, t;
        std::tie(m, s) = mean_std();
        t = sum();
        printf("time: mean %.2f us,  std %.2f us, total %.2f us\n", m, s, t);
    }
};

#endif /* TIMER_HPP */