#ifndef ABLATELIBRARY_UNION_HPP
#define ABLATELIBRARY_UNION_HPP

#include "geometry.hpp"

namespace ablate::mathFunctions::geom {

class Union : public Geometry {
   private:
    const std::vector<std::shared_ptr<ablate::mathFunctions::geom::Geometry>> geometries;

   public:
    explicit Union(std::vector<std::shared_ptr<ablate::mathFunctions::geom::Geometry>>  geometries, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {},
                   const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {});

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};
}  // namespace ablate::mathFunctions::geom

#endif  // ABLATELIBRARY_UNION_HPP
