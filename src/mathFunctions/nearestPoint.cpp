#include "nearestPoint.hpp"

ablate::mathFunctions::NearestPoint::NearestPoint(const std::vector<double> &coordinatesIn, const std::vector<double> &valuesIn)
    : coordinates(coordinatesIn), values(valuesIn), dimension(coordinatesIn.size() / valuesIn.size()), numberPoints(valuesIn.size()) {
    if (dimension < 1 || dimension > 3) {
        throw std::invalid_argument("The dimension (coordinates.size()/values.size()) must be 0, 1, or 3");
    }
}

std::size_t ablate::mathFunctions::NearestPoint::FindNearestPoint(const double *xyz, std::size_t xyzDimension) const {
    std::size_t index = 0;
    double nearestDistance = 1E10;

    // Determine the dimensions to check from the min of this class and xyz
    std::size_t dimensionToCheck = PetscMin(xyzDimension, dimension);
    // check each point
    for (std::size_t p = 0; p < numberPoints; ++p) {
        PetscReal distance = 0.0;

        for (std::size_t d = 0; d < dimensionToCheck; ++d) {
            distance += PetscSqr(xyz[d] - coordinates[p * dimension + d]);
        }
        distance = PetscSqrtReal(distance);
        if (distance < nearestDistance) {
            index = p;
            nearestDistance = distance;
        }
    }

    return index;
}

PetscErrorCode ablate::mathFunctions::NearestPoint::NearestPointPetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBegin;
    auto nearestPoint = (NearestPoint *)ctx;
    auto pt = nearestPoint->FindNearestPoint(x, dim);
    u[0] = nearestPoint->values[pt];
    PetscFunctionReturn(PETSC_SUCCESS);
}
double ablate::mathFunctions::NearestPoint::Eval(const double &x, const double &y, const double &z, const double &t) const {
    double xyz[3] = {x, y, z};
    auto pt = FindNearestPoint(xyz, 3);
    return values[pt];
}
double ablate::mathFunctions::NearestPoint::Eval(const double *xyz, const int &ndims, const double &t) const {
    auto pt = FindNearestPoint(xyz, ndims);
    return values[pt];
}
void ablate::mathFunctions::NearestPoint::Eval(const double &x, const double &y, const double &z, const double &t, std::vector<double> &result) const {
    double xyz[3] = {x, y, z};
    auto pt = FindNearestPoint(xyz, 3);
    if (result.size() != 1) {
        throw std::invalid_argument("The ablate::mathFunctions::NearestPoint only support scalar values");
    }
    result[0] = values[pt];
}
void ablate::mathFunctions::NearestPoint::Eval(const double *xyz, const int &ndims, const double &t, std::vector<double> &result) const {
    auto pt = FindNearestPoint(xyz, ndims);
    if (result.size() != 1) {
        throw std::invalid_argument("The ablate::mathFunctions::NearestPoint only support scalar values");
    }
    result[0] = values[pt];
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::NearestPoint, "Create a simple math function that initializes based upon the list of points",
         ARG(std::vector<double>, "coordinates", "list of coordinates (x1, y1, z1, x2, y2, etc.)"), ARG(std::vector<double>, "values", "list of values in the same order as the coordinates"));