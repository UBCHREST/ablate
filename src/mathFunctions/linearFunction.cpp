#include "linearFunction.hpp"
#include <algorithm>

ablate::mathFunctions::LinearFunction::LinearFunction(std::shared_ptr<MathFunction> startFunction, std::shared_ptr<MathFunction> endFunction, double start, double end, int dir)
    : startFunction(startFunction), endFunction(endFunction), start(start), end(end), dir(dir) {
    if (dir < 0 || dir > 2) {
        throw std::invalid_argument("The direction must be 0, 1, or 2 for x, y, or z");
    }
}
double ablate::mathFunctions::LinearFunction::Eval(const double& x, const double& y, const double& z, const double& t) const {
    auto startValue = startFunction->Eval(x, y, z, t);
    auto endValue = endFunction->Eval(x, y, z, t);

    return (Interpolate(DetermineDirectionValue(x, y, z), start, end, startValue, endValue));
}
double ablate::mathFunctions::LinearFunction::Eval(const double* xyz, const int& ndims, const double& t) const {
    auto startValue = startFunction->Eval(xyz, ndims, t);
    auto endValue = endFunction->Eval(xyz, ndims, t);
    return (Interpolate(xyz[dir], start, end, startValue, endValue));
}

void ablate::mathFunctions::LinearFunction::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    double xx = DetermineDirectionValue(x, y, z);

    std::vector<double> startValue(result.size());
    startFunction->Eval(x, y, z, t, startValue);

    std::vector<double> endValue(result.size());
    endFunction->Eval(x, y, z, t, endValue);

    for (std::size_t i = 0; i < std::min(result.size(), startValue.size()); ++i) {
        result[i] = Interpolate(xx, start, end, startValue[i], endValue[i]);
    }
}
void ablate::mathFunctions::LinearFunction::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    std::vector<double> startValue(result.size());
    startFunction->Eval(xyz, ndims, t, startValue);

    std::vector<double> endValue(result.size());
    endFunction->Eval(xyz, ndims, t, endValue);

    for (std::size_t i = 0; i < std::min(result.size(), startValue.size()); ++i) {
        result[i] = Interpolate(xyz[dir], start, end, startValue[i], endValue[i]);
    }
}
PetscErrorCode ablate::mathFunctions::LinearFunction::LinearFunctionPetscFunction(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt Nf, PetscScalar* u, void* ctx) {
    PetscFunctionBeginUser;
    auto linear = (ablate::mathFunctions::LinearFunction*)ctx;

    std::vector<double> startValue(Nf);
    linear->startFunction->Eval(x, dim, time, startValue);

    std::vector<double> endValue(Nf);
    linear->endFunction->Eval(x, dim, time, endValue);

    for (PetscInt i = 0; i < Nf; ++i) {
        u[i] = Interpolate(x[linear->dir], linear->start, linear->end, startValue[i], endValue[i]);
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::LinearFunction, "Linear interpolates between the start and end without extrapolation using other MathFunctions.",
         ARG(ablate::mathFunctions::MathFunction, "startFunction", "the start value(s)"), ARG(ablate::mathFunctions::MathFunction, "endFunction", "the end value(s)"),
         OPT(double, "start", "the start position"), OPT(double, "end", "the end position"), OPT(int, "dir", "the interpolation direction, 0 (default), 1, 2"));
