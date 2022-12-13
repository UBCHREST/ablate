#include "cylinderShell.hpp"

#include <utility>
#include "utilities/mathUtilities.hpp"

ablate::mathFunctions::geom::CylinderShell::CylinderShell(std::vector<double> start, std::vector<double> end, double radiusMin, double radiusMax,
                                                          const std::shared_ptr<mathFunctions::MathFunction> &insideValues, const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : Geometry(insideValues, outsideValues),
      start(std::move(start)),
      end(std::move(end)),
      radiusMin(radiusMin),
      radiusMax(radiusMax)

{}

bool ablate::mathFunctions::geom::CylinderShell::InsideGeometry(const double *xyz, const int &ndims, const double &) const {
    // Define vectors
    double dx[3];
    ablate::utilities::MathUtilities::Subtract(ndims, end.data(), start.data(), dx);
    double testDx[3];
    ablate::utilities::MathUtilities::Subtract(ndims, end.data(), xyz, testDx);

    // compute values
    double dot = ablate::utilities::MathUtilities::DotVector(ndims, dx, testDx);
    double length = ablate::utilities::MathUtilities::MagVector(ndims, dx);

    // easy end cap tests
    if (dot < 0.0 || dot > length * length) {
        return false;
    }

    // Now for the Hard Tests
    // distance squared to the cylinder axis
    double dsq = ablate::utilities::MathUtilities::DotVector(ndims, testDx, testDx) - dot * dot / (length * length);

    return !((dsq > (radiusMax * radiusMax)) || (radiusMin > 0.0 && dsq < (radiusMin * radiusMin)));
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::CylinderShell, "assigns a uniform value to all points inside a cylindrical shell",
         ARG(std::vector<double>, "start", "the center of the cylinder start"), ARG(std::vector<double>, "end", "the center of the cylinder end"),
         OPT(double, "radiusMin", "the cylinder shell inside radius"), OPT(double, "radiusMax", "the cylinder outside radius"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));
