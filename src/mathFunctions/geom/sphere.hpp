#ifndef ABLATELIBRARY_SPHERE_HPP
#define ABLATELIBRARY_SPHERE_HPP

#include "geometry.hpp"
namespace ablate::mathFunctions::geom {

class Sphere : public Geometry {
   private:
    const std::vector<double> center;
    const double radius;

   public:
    Sphere(std::vector<double> center, double radius, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {}, const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_SPHERE_HPP
