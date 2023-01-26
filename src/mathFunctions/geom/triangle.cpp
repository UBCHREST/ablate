#include "triangle.hpp"
#include <algorithm>
#include "utilities/constants.hpp"
#include "utilities/mathUtilities.hpp"

ablate::mathFunctions::geom::Triangle::Triangle(std::vector<double> point1In, std::vector<double> point2In, std::vector<double> point3In, double maxDistance,
                                                const std::shared_ptr<mathFunctions::MathFunction>& insideValues, const std::shared_ptr<mathFunctions::MathFunction>& outsideValues)
    : Geometry(insideValues, outsideValues), maxDistance(maxDistance) {
    // Make sure the input values sizes are correct
    if (!(point1In.size() == point2In.size() && point2In.size() == point3In.size())) {
        throw std::invalid_argument("All points must be same size in ablate::mathFunctions::geom::Triangle::Triangle");
    }

    if (point1In.size() < 2 || point1In.size() > 3) {
        throw std::invalid_argument("The dimension for a triangle must be 2 or 3");
    }

    // copy into the arrays
    std::copy_n(point1In.begin(), point1In.size(), point1.begin());
    std::copy_n(point2In.begin(), point2In.size(), point2.begin());
    std::copy_n(point3In.begin(), point3In.size(), point3.begin());

    // compute outward facing normals
    double p1_p2[3] = {0.0, 0.0, 0.0};
    double p2_p3[3] = {0.0, 0.0, 0.0};
    double p1_p3[3] = {0.0, 0.0, 0.0};
    ablate::utilities::MathUtilities::Subtract(3, point2.data(), point1.data(), p1_p2);
    ablate::utilities::MathUtilities::Subtract(3, point3.data(), point2.data(), p2_p3);
    ablate::utilities::MathUtilities::Subtract(3, point3.data(), point1.data(), p1_p3);

    ablate::utilities::MathUtilities::CrossVector<3>(p1_p2, p1_p3, triangleNorm.data());

    // Compute the side normals
    ablate::utilities::MathUtilities::CrossVector<3>(p1_p2, triangleNorm.data(), sideNorm3.data());
    ablate::utilities::MathUtilities::CrossVector<3>(p2_p3, triangleNorm.data(), sideNorm1.data());
    ablate::utilities::MathUtilities::CrossVector<3>(p1_p3, triangleNorm.data(), sideNorm2.data());
    ablate::utilities::MathUtilities::ScaleVector(3, sideNorm2.data(), -1.0);
}
bool ablate::mathFunctions::geom::Triangle::InsideGeometry(const double* xyz, const int& ndims, const double& time) const {
    // First check to see if it in the projected length
    double point[3] = {0.0, 0.0, 0.0};
    std::copy_n(xyz, ndims, point);

    // dot the
    // Get side normals
    double sign3 = ablate::utilities::MathUtilities::DiffDotVector<3>(point, point1.data(), sideNorm3.data());
    double sign2 = ablate::utilities::MathUtilities::DiffDotVector<3>(point, point3.data(), sideNorm2.data());
    double sign1 = ablate::utilities::MathUtilities::DiffDotVector<3>(point, point3.data(), sideNorm1.data());

    // Check to see if inside
    bool inside = (sign3 <= ablate::utilities::Constants::small) && (sign2 <= ablate::utilities::Constants::small) && (sign1 <= ablate::utilities::Constants::small);

    // check to see if we need to do a max distance check
    if (!inside) {
        return false;
    } else if (maxDistance == 0.0) {
        return true;
    }

    // Now check the max distance
    double disVecMag = ablate::utilities::MathUtilities::DiffDotVector<3>(point1.data(), point, triangleNorm.data()) / ablate::utilities::MathUtilities::MagVector(3, triangleNorm.data());
    return std::abs(disVecMag) < maxDistance;
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::geom::Geometry, ablate::mathFunctions::geom::Triangle, "Creates a 3D triangle including all projected space up to the maxDistance (on either side)",
         ARG(std::vector<double>, "point1", "the first point of the triangle"), ARG(std::vector<double>, "point2", "the second point of the triangle"),
         ARG(std::vector<double>, "point3", "the third point of the triangle"),
         OPT(double, "maxDistance", "max distance from triangle to be considered inside.  A zero value is assumed to include all projected space"),
         OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outsideValues, defaults to zero"));
