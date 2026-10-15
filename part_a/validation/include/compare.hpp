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
                           bool allow_masked_infinity = false) {
    if (actual.size() != expected.size()) throw std::invalid_argument("Comparison size mismatch");
    if (!std::isfinite(atol) || !std::isfinite(rtol) || atol<0 || rtol<0)
        throw std::invalid_argument("Invalid tolerance");
    ErrorReport report;
    for (std::size_t i=0; i<actual.size(); ++i) {
        const double a=actual[i], e=expected[i];
        if (allow_masked_infinity && std::isinf(e) && e<0 && a==e) continue;
        if (!std::isfinite(a) || !std::isfinite(e)) {
            ++report.failures;
            report.max_absolute = report.max_relative = std::numeric_limits<double>::infinity();
            continue;
        }
        const double error=std::abs(a-e);
        report.max_absolute=std::max(report.max_absolute,error);
        // Relative report; acceptance uses atol+rtol*abs(reference)
        report.max_relative=std::max(report.max_relative,error/std::max(std::abs(e),1e-12));
        if (error > atol + rtol*std::abs(e)) ++report.failures;
    }
    return report;
}
}
