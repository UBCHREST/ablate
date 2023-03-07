#ifndef ABLATELIBRARY_CONVEXPOLYGON_HPP
#define ABLATELIBRARY_CONVEXPOLYGON_HPP

#include <memory>
#include "geometry.hpp"
#include "triangle.hpp"
namespace ablate::mathFunctions::geom {

/**
 * Creates a convex polygon out of the center location and triangles.
 */

class ConvexPolygon : public Geometry {
   private:
    //! Keep a list of triangles used to represent this geometry
    std::vector<Triangle> triangles;

   public:
    explicit ConvexPolygon(std::vector<std::vector<double>> points, double maxDistance = {}, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {},
                           const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    /**
     * This is just helper constructor to because the registrar does not support vector of vectors
     * @param points
     * @param maxDistance
     * @param insideValues
     * @param outsideValues
     */
    explicit ConvexPolygon(std::vector<std::shared_ptr<std::vector<double>>> points, double maxDistance = {}, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {},
                           const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_CONVEXPOLYGON_HPP
