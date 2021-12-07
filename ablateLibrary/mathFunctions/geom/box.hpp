#ifndef ABLATELIBRARY_BOX_HPP
#define ABLATELIBRARY_BOX_HPP

#include "geometry.hpp"

namespace ablate::mathFunctions::geom {

class Box : public Geometry {
   private:
    const std::vector<double> lower;
    const std::vector<double> upper;

   public:
    Box(std::vector<double> lower, std::vector<double> upper, std::vector<double> insideValues = {}, std::vector<double> outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_SPHERE_HPP
