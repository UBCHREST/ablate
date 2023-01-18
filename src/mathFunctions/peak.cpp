#include "peak.hpp"
#include <algorithm>

ablate::mathFunctions::Peak::Peak(std::vector<double> startValue, std::vector<double> peakValue, std::vector<double> endValue, double start, double peak, double end, int dir)
    : startValue(startValue), peakValue(peakValue), endValue(endValue), start(start), peak(peak), end(end), dir(dir) {
    if (dir < 0 || dir > 2) {
        throw std::invalid_argument("The direction must be 0, 1, or 2 for x, y, or z");
    }

    if (startValue.empty() || endValue.empty() || peakValue.empty()) {
        throw std::invalid_argument("The start, end and peak values must not be empty.");
    }

    if (startValue.size() != endValue.size() || peakValue.size() != startValue.size()) {
        throw std::invalid_argument("The start, end, and peak values must be same size");
    }
}
double ablate::mathFunctions::Peak::Eval(const double& x, const double& y, const double& z, const double& t) const {
    return (Interpolate(DetermineDirectionValue(x, y, z), start, peak, end, startValue.front(), peakValue.front(), endValue.front()));
}
double ablate::mathFunctions::Peak::Eval(const double* xyz, const int& ndims, const double& t) const {
    return (Interpolate(xyz[dir], start, peak, end, startValue.front(), peakValue.front(), endValue.front()));
}
void ablate::mathFunctions::Peak::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    double xx = DetermineDirectionValue(x, y, z);

    for (std::size_t i = 0; i < std::min(result.size(), startValue.size()); ++i) {
        result[i] = Interpolate(xx, start, peak, end, startValue[i], peakValue[i], endValue[i]);
    }
}
void ablate::mathFunctions::Peak::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    for (std::size_t i = 0; i < std::min(result.size(), startValue.size()); ++i) {
        result[i] = Interpolate(xyz[dir], start, peak, end, startValue[i], peakValue[i], endValue[i]);
    }
}
PetscErrorCode ablate::mathFunctions::Peak::PeakPetscFunction(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt Nf, PetscScalar* u, void* ctx) {
    PetscFunctionBeginUser;
    auto peak = (ablate::mathFunctions::Peak*)ctx;

    for (PetscInt i = 0; i < PetscMin(Nf, (PetscInt)peak->startValue.size()); ++i) {
        u[i] = Interpolate(x[peak->dir], peak->start, peak->peak, peak->end, peak->startValue[i], peak->peakValue[i], peak->endValue[i]);
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::Peak, "Peak interpolates between the start, peak location, and end without extrapolation.",
         ARG(std::vector<double>, "startValues", "the start value(s)"), ARG(std::vector<double>, "peakValues", "the peak value(s)"), ARG(std::vector<double>, "endValues", "the end value(s)"),
         OPT(double, "start", "the start position"), OPT(double, "peak", "the peak position"), OPT(double, "end", "the end position"),
         OPT(int, "dir", "the interpolation direction, 0 (default), 1, 2"));
