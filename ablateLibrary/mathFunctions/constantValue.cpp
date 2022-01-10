#include "constantValue.hpp"

PetscErrorCode ablate::mathFunctions::ConstantValue::ConstantValuePetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto value = (const double *)ctx;
    for (PetscInt i = 0; i < nf; i++) {
        u[i] = value[i];
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::mathFunctions::ConstantValue::ConstantValueUniformPetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto value = (const double *)ctx;
    for (PetscInt i = 0; i < nf; i++) {
        u[i] = value[0];
    }

    PetscFunctionReturn(0);
}


ablate::mathFunctions::ConstantValue::ConstantValue(double value) : value(std::vector<double>{value}), uniformValue(true) {}

ablate::mathFunctions::ConstantValue::ConstantValue(std::vector<double> values) : value(values), uniformValue(false) {}

double ablate::mathFunctions::ConstantValue::Eval(const double &x, const double &y, const double &z, const double &t) const { return value[0]; }
double ablate::mathFunctions::ConstantValue::Eval(const double *xyz, const int &ndims, const double &t) const { return value[0]; }
void ablate::mathFunctions::ConstantValue::Eval(const double &x, const double &y, const double &z, const double &t, std::vector<double> &result) const {
    if (uniformValue) {
        for (std::size_t i = 0; i < result.size(); i++) {
            result[i] = value[0];
        }
    } else if (result.size() == value.size()) {
        for (std::size_t i = 0; i < value.size(); i++) {
            result[i] = value[i];
        }
    } else {
        throw std::invalid_argument("The function and result size do not match.");
    }
}
void ablate::mathFunctions::ConstantValue::Eval(const double *xyz, const int &ndims, const double &t, std::vector<double> &result) const {
    if (uniformValue) {
        for (std::size_t i = 0; i < result.size(); i++) {
            result[i] = value[0];
        }
    } else if (result.size() == value.size()) {
        for (std::size_t i = 0; i < value.size(); i++) {
            result[i] = value[i];
        }
    } else {
        throw std::invalid_argument("The function and result size do not match.");
    }
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::mathFunctions::MathFunction, ablate::mathFunctions::ConstantValue, "sets a constant value to all values in field", double);