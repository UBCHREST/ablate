#include "difference.hpp"

#include <utility>

ablate::mathFunctions::geom::Difference::Difference(std::shared_ptr<ablate::mathFunctions::geom::Geometry> minuend, std::shared_ptr<ablate::mathFunctions::geom::Geometry> subtrahend,
                                                    const std::shared_ptr<mathFunctions::MathFunction> &insideValues, const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : Geometry(insideValues, outsideValues), minuend(std::move(minuend)), subtrahend(std::move(subtrahend)) {}

bool ablate::mathFunctions::geom::Difference::InsideGeometry(const double *xyz, const int &ndims, const double &time) const {
    return minuend->InsideGeometry(xyz, ndims, time) && !subtrahend->InsideGeometry(xyz, ndims, time);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::Difference,
         "The geometry difference by the minuend - subtrahend. Note, this geometry ignores inside/outside values for base geometries.",
         ARG(ablate::mathFunctions::geom::Geometry, "minuend", "the minuend in minuend - subtrahend"),
         ARG(ablate::mathFunctions::geom::Geometry, "subtrahend", "the subtrahend in minuend - subtrahend"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));
