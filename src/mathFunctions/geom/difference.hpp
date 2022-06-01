#ifndef ABLATELIBRARY_DIFFERENCE_HPP
#define ABLATELIBRARY_DIFFERENCE_HPP

#include "geometry.hpp"

namespace ablate::mathFunctions::geom {

class Difference : public Geometry {
   private:
    const std::shared_ptr<ablate::mathFunctions::geom::Geometry> minuend;
    const std::shared_ptr<ablate::mathFunctions::geom::Geometry> subtrahend;

   public:
    explicit Difference(std::shared_ptr<ablate::mathFunctions::geom::Geometry>  minuend, std::shared_ptr<ablate::mathFunctions::geom::Geometry>  subtrahend,
                        const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {}, const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};
}  // namespace ablate::mathFunctions::geom

#endif  // ABLATELIBRARY_DIFFERENCE_HPP
