#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
namespace attention {
struct Distribution { double q1, median, q3; double iqr() const { return q3 - q1; } };
// Type-7 quantiles at (n-1)*p
inline Distribution distribution(std::vector<double> values) {
    if (values.empty()) throw std::invalid_argument("Empty timing samples");
    for (double value : values)
        if (!std::isfinite(value) || value <= 0) throw std::invalid_argument("Invalid timing sample");
    std::sort(values.begin(), values.end());
    auto quantile = [&](double p) {
        const double position = (values.size() - 1) * p;
        const auto lower = static_cast<std::size_t>(position);
        const auto upper = std::min(lower + 1, values.size() - 1);
        return values[lower] + (position - lower) * (values[upper] - values[lower]);
    };
    return {quantile(0.25), quantile(0.5), quantile(0.75)};
}
}
