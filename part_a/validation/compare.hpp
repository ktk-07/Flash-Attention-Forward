#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
namespace attention {
struct ErrorReport {
    std::size_t failures = 0;
    double max_absolute = 0, max_relative = 0;
    bool passed() const { return failures == 0; }
};

inline ErrorReport compare(const std::vector<float>& actual,
                           const std::vector<double>& expected,
                           double atol = 1e-4, double rtol = 1e-4,
                           bool masked_scores = false) {
    if (actual.size() != expected.size()) throw std::invalid_argument("Comparison size mismatch");
    ErrorReport report;
    for (std::size_t i=0; i<actual.size(); ++i) {
        const double a=actual[i], e=expected[i];
        // masked scores can be -inf
        if (masked_scores && std::isinf(e) && e<0 && a==e) continue;
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++report.failures;
            report.max_absolute = report.max_relative = std::numeric_limits<double>::infinity();
            continue;
        }

        // absolute and relative error
        const double error=std::abs(a-e);
        report.max_absolute=std::max(report.max_absolute,error);
        report.max_relative=std::max(report.max_relative,error/std::max(std::abs(e),1e-12));
        if (error > atol + rtol*std::abs(e)) ++report.failures;
    }
    return report;
}
}
