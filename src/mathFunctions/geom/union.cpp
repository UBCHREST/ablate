#include "union.hpp"

#include <utility>

ablate::mathFunctions::geom::Union::Union(std::vector<std::shared_ptr<ablate::mathFunctions::geom::Geometry>> geometries, const std::shared_ptr<mathFunctions::MathFunction> &insideValues,
                                          const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : Geometry(insideValues, outsideValues), geometries(std::move(geometries)) {}

bool ablate::mathFunctions::geom::Union::InsideGeometry(const double *xyz, const int &ndims, const double &time) const {
    return std::any_of(geometries.begin(), geometries.end(), [&](const auto &geometry) { return geometry->InsideGeometry(xyz, ndims, time); });
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::geom::Union,
         "Merges multiple geometries into a single geometry.  Note, this geometry ignores inside/outside values for unioned geometries.",
         ARG(std::vector<ablate::mathFunctions::geom::Geometry>, "geometries", "the geometries to be merged"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));

REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::Union,
         "Merges multiple geometries into a single geometry.  Note, this geometry ignores inside/outside values for unioned geometries.",
         ARG(std::vector<ablate::mathFunctions::geom::Geometry>, "geometries", "the geometries to be merged"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));