#ifndef ABLATELIBRARY_PARTICLES_HPP
#define ABLATELIBRARY_PARTICLES_HPP

#include <memory>
#include "flow/flow.hpp"
#include "mathFunctions/fieldFunction.hpp"
#include "mathFunctions/mathFunction.hpp"
#include "particles/initializers/initializer.hpp"
#include "particles/particleFieldDescriptor.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::particles {

class Particles : public io::Serializable, public monitors::Monitorable {
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
    Vec flowInitial; /* The PDE solution field at ti */
    Vec flowFinal;   /* The PDE solution field at tf */

    // all fields stored in the particle domain
    std::vector<particles::ParticleFieldDescriptor> particleFieldDescriptors;
    std::vector<particles::ParticleFieldDescriptor> particleSolutionDescriptors;

    // store the exact solution if provided
    const std::shared_ptr<mathFunctions::MathFunction> exactSolution;

    /**
     * The register fields adds the field to the swarm
     * @param fieldDescriptor
     */
    void RegisterField(ParticleFieldDescriptor fieldDescriptor);

    /**
     * The register solution fields adds the field to the swarm and includes the value in the pack/unpack solution
     * @param fieldDescriptor
     */
    void RegisterSolutionField(ParticleFieldDescriptor fieldDescriptor);

    // Petsc options specific to these particles. These may be null by default
    PetscOptions petscOptions;

    // store a boolean to state if a dmChanged (number of particles local/global changed)
    bool dmChanged;
    void SwarmMigrate();

    // Store the particle location and field initialization
    std::shared_ptr<particles::initializers::Initializer> initializer = nullptr;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization;

    /**
     * Gets and packs the solution vector
     * @return
     */
    Vec GetPackedSolutionVector();
    /**
     * Unpacks and returns the solution vector
     */
    void RestorePackedSolutionVector(Vec);

    /**
     * Gets and packs the solution vector
     * @return
     */
    void UnpackSolutionVector(Vec, std::vector<Vec>);

    /**
     * Function to be be called after each flow time step
     */
    void AdvectParticles(TS flowTS);

    /**
     * Get the name of the solution vector
     */
    inline const char* GetSolutionVectorName() { return particleSolutionDescriptors.size() == 1 ? DMSwarmPICField_coor : PackedSolution; }

   private:
    inline static const char PackedSolution[] = "PackedSolution";
    inline static const char ParticleInitialLocation[] = "InitialLocation";

    void StoreInitialParticleLocations();
    static PetscErrorCode ComputeParticleError(TS particleTS, Vec u, Vec e);

   public:
    explicit Particles(std::string name, int ndims, std::shared_ptr<particles::initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                       std::shared_ptr<mathFunctions::MathFunction> exactSolution, std::shared_ptr<parameters::Parameters> options);
    virtual ~Particles();

    const std::string& GetName() const override { return name; }
    const std::string& GetId() const override { return name; }
    const DM& GetDM() const { return dm; }
    PetscReal GetInitialTime() const { return timeInitial; }
    PetscReal GetFinalTime() const { return timeFinal; }
    TS GetTS() const { return particleTs; }

    virtual void InitializeFlow(std::shared_ptr<flow::Flow> flow);

    void ProjectFunction(const std::string& field, ablate::mathFunctions::MathFunction& mathFunction);

    /**
     * shared function to view all particles;
     * @param viewer
     * @param steps
     * @param time
     * @param u
     */
    void Save(PetscViewer viewer, PetscInt steps, PetscReal time) const override;

    /**
     * shared function to view all particles;
     * @param viewer
     * @param steps
     * @param time
     * @param u
     */
    void Restore(PetscViewer viewer, PetscInt steps, PetscReal time) override;

    /** common field names for particles **/
    inline static const char ParticleVelocity[] = "ParticleVelocity";
    inline static const char ParticleDiameter[] = "ParticleDiameter";
    inline static const char ParticleDensity[] = "ParticleDensity";

    // Helper function useful for tests
    static PetscErrorCode ComputeParticleExactSolution(TS particleTS, Vec);
};
}  // namespace ablate::particles
#endif  // ABLATELIBRARY_PARTICLES_HPP
