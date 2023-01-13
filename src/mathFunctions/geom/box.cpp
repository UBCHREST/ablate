#include "box.hpp"
#include <petsc.h>
ablate::mathFunctions::geom::Box::Box(std::vector<double> lower, std::vector<double> upper, const std::shared_ptr<mathFunctions::MathFunction> &insideValues,
                                      const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : Geometry(insideValues, outsideValues), lower(lower), upper(upper) {
    if (lower.size() != upper.size()) {
        throw std::invalid_argument("The lower and upper bounds in ablate::geom::Box must be the same size.");
    }
}

bool ablate::mathFunctions::geom::Box::InsideGeometry(const double *xyz, const int &ndims, const double &) const {
    for (std::size_t i = 0; i < PetscMin((std::size_t)ndims, lower.size()); i++) {
        if (xyz[i] < lower[i] || xyz[i] > upper[i]) {
            return false;
        }
    }
    return true;
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::Box, "assigns a uniform value to all points inside the box", ARG(std::vector<double>, "lower", "the box lower corner"),
         ARG(std::vector<double>, "upper", "the box upper corner"), OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));
