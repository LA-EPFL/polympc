#ifndef TIMER_HPP
#define TIMER_HPP

#include <stdio.h>

#include <chrono>
#include <vector>
#include <tuple>
#include <numeric>
#include <algorithm>

template <typename Scalar = double>
class Timer {
    /** OS dependent */
#ifdef __APPLE__
    using clock = std::chrono::system_clock;
#else
    using clock = std::chrono::high_resolution_clock;
#endif
    using time_point = std::chrono::time_point<clock>;

    time_point _start, _stop;
    std::vector<Scalar> _samples;

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
        Scalar t = std::chrono::duration<Scalar, std::micro>(_stop - _start).count();
        _samples.push_back(t);
    }

    void clear()
    {
        _samples.clear();
    }

    const std::vector<Scalar>& samples()
    {
        return _samples;
    }

    Scalar sum()
    {
        return std::accumulate(_samples.begin(), _samples.end(), 0.0);
    }

    Scalar mean()
    {
        if (_samples.size() == 0) {
            return 0;
        }
        return sum() / _samples.size();
    }

    std::tuple<Scalar, Scalar> mean_std()
    {
        Scalar m, s;

        if (_samples.size() == 0) {
            return std::make_tuple(0.0, 0.0);
        }

        m = mean();

        std::vector<Scalar> diff(_samples.size());
        std::transform(_samples.begin(), _samples.end(), diff.begin(),
                       [m](Scalar x) {
            return x - m;
        }
                       );
        Scalar sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
        s = sqrt(sq_sum / diff.size());

        return std::make_tuple(m, s);
    }

    void print()
    {
        double m, s;
        std::tie(m, s) = mean_std();
        printf("time: mean %.2f us,  std %.2f us\n", m, s);
    }
};

#endif /* TIMER_HPP */