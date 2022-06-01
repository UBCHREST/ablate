#include "cylinder.hpp"

#include <utility>

ablate::mathFunctions::geom::Cylinder::Cylinder(std::vector<double> start, std::vector<double> end, double radius, const std::shared_ptr<mathFunctions::MathFunction> &insideValues,
                                                const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : CylinderShell(std::move(start), std::move(end), 0.0, radius, insideValues, outsideValues)

{}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::geom::Cylinder, "assigns a uniform value to all points inside a cylinder",
         ARG(std::vector<double>, "start", "the center of the cylinder start"), ARG(std::vector<double>, "end", "the center of the cylinder end"), OPT(double, "radius", "the cylinder outside radius"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));

REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::Cylinder, "assigns a uniform value to all points inside a cylinder",
         ARG(std::vector<double>, "start", "the center of the cylinder start"), ARG(std::vector<double>, "end", "the center of the cylinder end"), OPT(double, "radius", "the cylinder outside radius"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));