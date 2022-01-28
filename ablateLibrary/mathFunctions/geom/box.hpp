#ifndef ABLATELIBRARY_BOX_HPP
#define ABLATELIBRARY_BOX_HPP

#include "geometry.hpp"

namespace ablate::mathFunctions::geom {

class Box : public Geometry {
   private:
    const std::vector<double> lower;
    const std::vector<double> upper;

   public:
    Box(std::vector<double> lower, std::vector<double> upper, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {},
        const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_SPHERE_HPP
