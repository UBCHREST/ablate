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
#include "der.hpp"



namespace ablate::levelSet {


class LevelSetField : public domain::FieldDescriptor {

  public:
    enum class levelSetShape {SPHERE, ELLIPSE, STAR};

    // Constructors
    LevelSetField(std::shared_ptr<domain::Region> = {});
    LevelSetField(std::shared_ptr<RBF> rbf = {}, levelSetShape shape = LevelSetField::levelSetShape::SPHERE);

    // Destructor
    ~LevelSetField();

    // Copied from current ABLATE code. Need to talk to Matt M. about how to integrate
    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;

    // Return the vector containing the level set vector
    inline Vec& GetVec() noexcept { return phi; }

    // Return the mesh associated with the level set
    inline DM& GetDM() noexcept { return dm; }

    PetscReal Curvature(PetscInt c);

   private:
    // Copied from current ABLATE code. Need to talk to Matt M. about how to integrate
    const std::shared_ptr<domain::Region> region = nullptr;

    // Internal curvature and normal calculations
    PetscReal Curvature2D(PetscInt c);
    PetscReal Curvature3D(PetscInt c);

    // The RBF to be used for derivatives
    std::shared_ptr<RBF> rbf = nullptr;

    // The derivative class
    std::shared_ptr<DerCalculator> der = nullptr;

    // Possible initial shapes
    PetscReal Sphere(PetscReal pos[], PetscReal center[], PetscReal radius);
    PetscReal Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius);
    PetscReal Star(PetscReal pos[], PetscReal center[]);

    // The level set data
    Vec phi = nullptr;

    // Underlying mesh
    DM dm = nullptr;
};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_LEVELSETFIELDS_HPP




