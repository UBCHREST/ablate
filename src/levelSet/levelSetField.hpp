#ifndef ABLATELIBRARY_LEVELSETFIELDS_HPP
#define ABLATELIBRARY_LEVELSETFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <cstring>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "parameters/mapParameters.hpp"
#include "domain/fieldDescription.hpp"
#include "utilities/petscError.hpp"
#include "rbf.hpp"



namespace ablate::levelSet {


class LevelSetField : public domain::FieldDescriptor {

  public:
    enum class levelSetShape {SPHERE, ELLIPSE, STAR};
    LevelSetField(std::shared_ptr<domain::Region> = {});
    LevelSetField(std::shared_ptr<RBF> rbf = {}, levelSetShape shape = LevelSetField::levelSetShape::SPHERE);

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;

    // Return the vector containing the level set vector
    inline Vec& GetVec() noexcept { return phi; }

    // Return the mesh associated with the level set
    inline DM& GetDM() noexcept { return dm; }


   private:
    const std::shared_ptr<domain::Region> region = nullptr;
    std::shared_ptr<RBF> rbf = nullptr;
    PetscReal Sphere(PetscReal pos[], PetscReal center[], PetscReal radius);
    PetscReal Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius);
    PetscReal Star(PetscReal pos[], PetscReal center[]);
    Vec phi = nullptr;
    DM dm = nullptr;
};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_LEVELSETFIELDS_HPP




