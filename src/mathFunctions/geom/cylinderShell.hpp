#ifndef ABLATELIBRARY_CYLINDERSHELL_HPP
#define ABLATELIBRARY_CYLINDERSHELL_HPP

#include "geometry.hpp"
namespace ablate::mathFunctions::geom {

class CylinderShell : public Geometry {
   private:
    const std::vector<double> start;
    const std::vector<double> end;
    const double radiusMin;
    const double radiusMax;

   public:
    CylinderShell(std::vector<double> start, std::vector<double> end, double radiusMin, double radiusMax, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {},
                  const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_SPHERE_HPP
