#include "geometry.hpp"

ablate::mathFunctions::geom::Geometry::Geometry(std::vector<double> insideValues, std::vector<double> outsideValuesIn)
    : insideValues(insideValues), outsideValues(outsideValuesIn.empty() ? std::vector<double>(insideValues.size(), 0.0) : outsideValuesIn) {}

PetscErrorCode ablate::mathFunctions::geom::Geometry::GeometryPetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto geom = (ablate::mathFunctions::geom::Geometry *)ctx;

    if (geom->InsideGeometry(x, dim, time)) {
        if (nf == (PetscInt)geom->insideValues.size()) {
            for (std::size_t i = 0; i < geom->insideValues.size(); i++) {
                u[i] = geom->insideValues[i];
            }
        } else {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "The function and result size do not match.");
        }
    } else {
        if (nf == (PetscInt)geom->outsideValues.size()) {
            for (std::size_t i = 0; i < geom->outsideValues.size(); i++) {
                u[i] = geom->outsideValues[i];
            }
        } else {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "The function and result size do not match.");
        }
    }

    PetscFunctionReturn(0);
}
double ablate::mathFunctions::geom::Geometry::Eval(const double &x, const double &y, const double &z, const double &t) const {
    double temp[3] = {x, y, z};

    return InsideGeometry(temp, 3, t) ? insideValues[0] : outsideValues[0];
}
double ablate::mathFunctions::geom::Geometry::Eval(const double *xyz, const int &ndims, const double &t) const { return InsideGeometry(xyz, 3, t) ? insideValues[0] : outsideValues[0]; }
void ablate::mathFunctions::geom::Geometry::Eval(const double &x, const double &y, const double &z, const double &t, std::vector<double> &result) const {
    double temp[3] = {x, y, z};
    if (InsideGeometry(temp, 3, t)) {
        if (result.size() == insideValues.size()) {
            for (std::size_t i = 0; i < insideValues.size(); i++) {
                result[i] = insideValues[i];
            }
        } else {
            throw std::invalid_argument("The function and result size do not match.");
        }
    } else {
        if (result.size() == outsideValues.size()) {
            for (std::size_t i = 0; i < outsideValues.size(); i++) {
                result[i] = outsideValues[i];
            }
        } else {
            throw std::invalid_argument("The function and result size do not match.");
        }
    }
}
void ablate::mathFunctions::geom::Geometry::Eval(const double *xyz, const int &ndims, const double &t, std::vector<double> &result) const {
    if (InsideGeometry(xyz, 3, t)) {
        if (result.size() == insideValues.size()) {
            for (std::size_t i = 0; i < insideValues.size(); i++) {
                result[i] = insideValues[i];
            }
        } else {
            throw std::invalid_argument("The function and result size do not match.");
        }
    } else {
        if (result.size() == outsideValues.size()) {
            for (std::size_t i = 0; i < outsideValues.size(); i++) {
                result[i] = outsideValues[i];
            }
        } else {
            throw std::invalid_argument("The function and result size do not match.");
        }
    }
}
