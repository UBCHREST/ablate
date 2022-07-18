#include "inverse.hpp"

ablate::mathFunctions::geom::Inverse::Inverse(const std::shared_ptr<ablate::mathFunctions::geom::Geometry> &geometry)
    : Geometry(geometry->InsideValues(), geometry->OutsideValues()), geometry(geometry) {}

bool ablate::mathFunctions::geom::Inverse::InsideGeometry(const double *xyz, const int &ndims, const double &time) const { return !geometry->InsideGeometry(xyz, ndims, time); }

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::geom::Inverse, "Inverses the supplied geometry.",
         ARG(ablate::mathFunctions::geom::Geometry, "geometry", "the base geometry to be inversed"));

REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::Inverse, "Inverses the supplied geometry.",
         ARG(ablate::mathFunctions::geom::Geometry, "geometry", "the base geometry to be inversed"));