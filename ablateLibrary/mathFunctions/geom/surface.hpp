#ifndef ABLATELIBRARY_GEOM_SURFACE_HPP
#define ABLATELIBRARY_GEOM_SURFACE_HPP

#include <egads.h>
#include <petsc.h>
#include <filesystem>
#include "geometry.hpp"

namespace ablate::mathFunctions::geom {

class Surface : public Geometry {
   private:
    ego context = nullptr;
    ego model = nullptr;

   public:
    explicit Surface(std::filesystem::path meshPath, const std::shared_ptr<mathFunctions::MathFunction>& insideValues = {}, const std::shared_ptr<mathFunctions::MathFunction>& outsideValues = {},
                     int egadsVerboseLevel = 0);
    ~Surface() override;

    bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const override;
};
}  // namespace ablate::mathFunctions::geom

#endif  // ABLATELIBRARY_MESH_HPP
