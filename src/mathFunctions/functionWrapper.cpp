#include "functionWrapper.hpp"
#include <array>

ablate::mathFunctions::FunctionWrapper::FunctionWrapper(ablate::mathFunctions::FunctionWrapper::Function function) : function(function) {}

double ablate::mathFunctions::FunctionWrapper::Eval(const double& x, const double& y, const double& z, const double& t) const {
    std::array<double, 3> loc = {x, y, z};
    double result;
    function(loc.size(), t, &loc[0], 1, &result, nullptr);
    return result;
}

double ablate::mathFunctions::FunctionWrapper::Eval(const double* xyz, const int& ndims, const double& t) const {
    double result;
    function(ndims, t, xyz, 1, &result, nullptr);
    return result;
}

void ablate::mathFunctions::FunctionWrapper::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    std::array<double, 3> loc = {x, y, z};
    function(3, t, &loc[0], result.size(), &result[0], nullptr);
}

void ablate::mathFunctions::FunctionWrapper::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    function(ndims, t, xyz, result.size(), &result[0], nullptr);
}

PetscErrorCode ablate::mathFunctions::FunctionWrapper::WrappedPetscFunction(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt Nf, PetscScalar* u, void* ctx) {
    auto function = (Function*)ctx;
    return (*function)(dim, time, x, Nf, u, nullptr);
}
