#ifndef ABLATELIBRARY_PARTICLES_HPP
#define ABLATELIBRARY_PARTICLES_HPP

#include <memory>
#include "flow/flow.hpp"
#include "mathFunctions/mathFunction.hpp"
#include "particles/initializers/initializer.hpp"
#include "solve/timeStepper.hpp"
#include "particles/particleFieldDescriptor.hpp"

namespace ablate::particles {

class Particles {
   protected:
    // particle domain
    DM dm;
    const PetscInt ndims;

    // particle name
    const std::string name;

    // time integration data
    TS particleTs = NULL;
    PetscReal timeInitial; /* The time for ui, at the beginning of the advection solve */
    PetscReal timeFinal;   /* The time for uf, at the end of the advection solve */

    // flow coupling data
    PetscInt flowVelocityFieldIndex;
    Vec flowInitial;       /* The PDE solution field at ti */
    Vec flowFinal;         /* The PDE solution field at tf */

    // all fields stored in the particle domain
    std::vector<particles::ParticleFieldDescriptor> particleFieldDescriptors;

    void RegisterField(ParticleFieldDescriptor fieldDescriptor);
    std::shared_ptr<particles::initializers::Initializer> initializer = nullptr;

    // Petsc options specific to these particles. These may be null by default
    PetscOptions petscOptions;

    // store a boolean to state if a dmChanged (number of particles local/global changed)
    bool dmChanged;

   private:
    inline static const char ParticleInitialLocation[] = "InitialLocation";
    void StoreInitialParticleLocations();
    static PetscErrorCode ComputeParticleError(TS particleTS, Vec u, Vec e);

   public:
    explicit Particles(std::string name,  int ndims, std::shared_ptr<particles::initializers::Initializer> initializer,std::shared_ptr<mathFunctions::MathFunction> exactSolution, std::shared_ptr<parameters::Parameters> options);
    virtual ~Particles();

    const std::string& GetName() const { return name; }
    const DM& GetDM() const { return dm; }
    PetscReal GetInitialTime() const { return timeInitial; }
    PetscReal GetFinalTime() const { return timeFinal; }
    const TS GetTS() const { return particleTs;}

    virtual void InitializeFlow(std::shared_ptr<flow::Flow> flow);

    const std::shared_ptr<mathFunctions::MathFunction> exactSolution = nullptr;
};
}  // namespace ablate::particles
#endif  // ABLATELIBRARY_PARTICLES_HPP
