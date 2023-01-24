#include "linear.hpp"
#include <algorithm>

ablate::mathFunctions::Linear::Linear(std::vector<double> startValue, std::vector<double> endValue, double start, double end, int dir)
    : startValue(startValue), endValue(endValue), start(start), end(end), dir(dir) {
    if (dir < 0 || dir > 2) {
        throw std::invalid_argument("The direction must be 0, 1, or 2 for x, y, or z");
    }

    if (startValue.empty() || endValue.empty()) {
        throw std::invalid_argument("The start and end values must not be empty.");
    }

    if (startValue.size() != endValue.size()) {
        throw std::invalid_argument("The start and end values must be same size");
    }
}
double ablate::mathFunctions::Linear::Eval(const double& x, const double& y, const double& z, const double& t) const {
    return (Interpolate(DetermineDirectionValue(x, y, z), start, end, startValue.front(), endValue.front()));
}
double ablate::mathFunctions::Linear::Eval(const double* xyz, const int& ndims, const double& t) const { return (Interpolate(xyz[dir], start, end, startValue.front(), endValue.front())); }
void ablate::mathFunctions::Linear::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    double xx = DetermineDirectionValue(x, y, z);

    for (std::size_t i = 0; i < std::min(result.size(), startValue.size()); ++i) {
        result[i] = Interpolate(xx, start, end, startValue[i], endValue[i]);
    }
}
void ablate::mathFunctions::Linear::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    for (std::size_t i = 0; i < std::min(result.size(), startValue.size()); ++i) {
        result[i] = Interpolate(xyz[dir], start, end, startValue[i], endValue[i]);
    }
}
PetscErrorCode ablate::mathFunctions::Linear::LinearPetscFunction(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt Nf, PetscScalar* u, void* ctx) {
    PetscFunctionBeginUser;
    auto linear = (ablate::mathFunctions::Linear*)ctx;

    for (PetscInt i = 0; i < PetscMin(Nf, (PetscInt)linear->startValue.size()); ++i) {
        u[i] = Interpolate(x[linear->dir], linear->start, linear->end, linear->startValue[i], linear->endValue[i]);
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::Linear, "Linear interpolates between the start and end without extrapolation.",
         ARG(std::vector<double>, "startValues", "the start value(s)"), ARG(std::vector<double>, "endValues", "the end value(s)"), OPT(double, "start", "the start position"),
         OPT(double, "end", "the end position"), OPT(int, "dir", "the interpolation direction, 0 (default), 1, 2"));
