#ifndef ABLATELIBRARY_TRIANGLE_HPP
#define ABLATELIBRARY_TRIANGLE_HPP

#include <array>
#include "cylinderShell.hpp"

namespace ablate::mathFunctions::geom {

/**
 * Geometry to determine if a point is inside of a triangular prism (projected from both sides of triangle). An optional max distance from the base triangle can be provided.
 */
class Triangle : public Geometry {
   private:
    /**
     * The three points of the triangle
     */
    std::array<double, 3> point1 = {0, 0, 0};
    std::array<double, 3> point2 = {0, 0, 0};
    std::array<double, 3> point3 = {0, 0, 0};

    //! max distance from triangle to be considered inside.  A zero value is assumed to include all projected space
    const double maxDistance;

    /**
     * Precompute some of the needed values
     */
    std::array<double, 3> sideNorm1 = {0, 0, 0};
    std::array<double, 3> sideNorm2 = {0, 0, 0};
    std::array<double, 3> sideNorm3 = {0, 0, 0};
    std::array<double, 3> triangleNorm = {0, 0, 0};

   public:
    Triangle(std::vector<double> pt0, std::vector<double> pt1, std::vector<double> pt2, double maxDistance = {}, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {},
             const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_TRIANGLE_HPP
