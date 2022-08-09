#ifndef ABLATELIBRARY_INVERSE_HPP
#define ABLATELIBRARY_INVERSE_HPP

#include "geometry.hpp"

namespace ablate::mathFunctions::geom {

class Inverse : public Geometry {
   private:
    const std::shared_ptr<ablate::mathFunctions::geom::Geometry> geometry;

   public:
    explicit Inverse(const std::shared_ptr<ablate::mathFunctions::geom::Geometry>& geometry);

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};
}  // namespace ablate::mathFunctions::geom

#endif  // ABLATELIBRARY_INVERSE_HPP
