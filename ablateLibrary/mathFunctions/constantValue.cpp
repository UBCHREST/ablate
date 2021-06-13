#include "constantValue.hpp"

PetscErrorCode ablate::mathFunctions::ConstantValue::ConstantValuePetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto value = (const std::vector<double> *)ctx;

    if (nf == value->size()) {
        for (auto i = 0; i < value->size(); i++) {
            u[i] = (*value)[i];
        }
    } else {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "The function and result size do not match.");
    }

    PetscFunctionReturn(0);
}

ablate::mathFunctions::ConstantValue::ConstantValue(double value) : ConstantValue(std::vector<double>{value}) {}

ablate::mathFunctions::ConstantValue::ConstantValue(std::vector<double> values) : value(values) {}

double ablate::mathFunctions::ConstantValue::Eval(const double &x, const double &y, const double &z, const double &t) const { return value[0]; }
double ablate::mathFunctions::ConstantValue::Eval(const double *xyz, const int &ndims, const double &t) const { return value[0]; }
void ablate::mathFunctions::ConstantValue::Eval(const double &x, const double &y, const double &z, const double &t, std::vector<double> &result) const {
    if (result.size() == value.size()) {
        for (auto i = 0; i < value.size(); i++) {
            result[i] = value[i];
        }
    } else {
        throw std::invalid_argument("The function and result size do not match.");
    }
}
void ablate::mathFunctions::ConstantValue::Eval(const double *xyz, const int &ndims, const double &t, std::vector<double> &result) const {
    if (result.size() == value.size()) {
        for (auto i = 0; i < value.size(); i++) {
            result[i] = value[i];
        }
    } else {
        throw std::invalid_argument("The function and result size do not match.");
    }
}
