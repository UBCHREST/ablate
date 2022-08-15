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

    // Constructors
    LevelSetField(std::shared_ptr<domain::Region> = {});
    LevelSetField(std::shared_ptr<RBF> rbf = {}, levelSetShape shape = LevelSetField::levelSetShape::SPHERE);

    // Destructor
    ~LevelSetField();

    // Copied from current ABLATE code. Need to talk to Matt M. about how to integrate
    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;

    // Return the vectors
    inline Vec& GetPhi() noexcept { return phi; }
    inline Vec& GetCurv() noexcept { return curv; }
    inline Vec& GetNormal() noexcept { return normal; }

    // Return the mesh associated with the level set
    inline DM& GetDM() noexcept { return dm; }

    PetscReal Curvature(PetscInt c);
    void Normal(PetscInt c, PetscReal *n);

    void ComputeAllCurvature();
    void ComputeAllNormal();

    // Level set function interpolation
    PetscReal Interpolate(const PetscReal x, const double y, const double z);
    PetscReal Interpolate(PetscReal xyz[3]);

    // Given a velocity field advect the level set
    void Advect(Vec vel, const PetscReal dt);

    bool HasInterface(const PetscInt p);

    PetscReal VOF(const PetscInt c);

   private:
    // Copied from current ABLATE code. Need to talk to Matt M. about how to integrate
    const std::shared_ptr<domain::Region> region = nullptr;

    // Internal curvature and normal calculations
    PetscReal Curvature2D(PetscInt c);
    PetscReal Curvature3D(PetscInt c);
    void Normal2D(PetscInt c, PetscReal *n);
    void Normal3D(PetscInt c, PetscReal *n);

    // The RBF to be used for derivatives
    std::shared_ptr<RBF> rbf = nullptr;

    // Possible initial shapes
    PetscReal Sphere(PetscReal pos[], PetscReal center[], PetscReal radius);
    PetscReal Ellipse(PetscReal pos[], PetscReal center[], PetscReal radius);
    PetscReal Star(PetscReal pos[], PetscReal center[]);

    // The level set data
    Vec phi = nullptr;

    // The curvature
    Vec curv = nullptr;

    // The unit normal
    Vec normal = nullptr;

    // Underlying mesh
    DM dm = nullptr;

    PetscInt dim;
};

}  // namespace ablate::levelSet

#endif  // ABLATELIBRARY_LEVELSETFIELDS_HPP




