#ifndef ABLATELIBRARY_CYLINDER_HPP
#define ABLATELIBRARY_CYLINDER_HPP

#include "cylinderShell.hpp"
namespace ablate::mathFunctions::geom {

class Cylinder : public CylinderShell {
   public:
    Cylinder(std::vector<double> start, std::vector<double> end, double radius, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {},
             const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_SPHERE_HPP
