#ifndef ABLATELIBRARY_PARTICLES_HPP
#define ABLATELIBRARY_PARTICLES_HPP

#include <memory>
#include "mathFunctions/fieldFunction.hpp"
#include "mathFunctions/mathFunction.hpp"
#include "particleField.hpp"
#include "particles/initializers/initializer.hpp"
#include "solver/solver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::particles {

class Particles : public solver::Solver {
   protected:
    // particle domain
    DM swarmDm;
    const PetscInt ndims;

    // time integration data
    TS particleTs = NULL;
    PetscReal timeInitial; /* The time for ui, at the beginning of the advection solve */
    PetscReal timeFinal;   /* The time for uf, at the end of the advection solve */

    // flow coupling data
    Vec flowInitial; /* The PDE solution field at ti */
    Vec flowFinal;   /* The PDE solution field at tf */

    // all fields stored in the particle domain
    std::vector<ParticleField> particleFieldDescriptors;
    std::vector<ParticleField> particleSolutionFieldDescriptors;

    // store the exact solution if provided
    const std::shared_ptr<mathFunctions::MathFunction> exactSolution;

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
    inline const char* GetSolutionVectorName() { return particleSolutionFieldDescriptors.size() == 1 ? DMSwarmPICField_coor : PackedSolution; }

    /**
     * Support function to create a vector of strings
     */
    static std::vector<std::string> CreateDimensionVector(const std::string& prefix, int dim);

   private:
    inline static const char PackedSolution[] = "PackedSolution";
    inline static const char ParticleInitialLocation[] = "InitialLocation";

    void StoreInitialParticleLocations();
    static PetscErrorCode ComputeParticleError(TS particleTS, Vec u, Vec e);

    /**
     * The register fields adds the field to the swarm
     * @param fieldDescriptor
     */
    void RegisterParticleField(const ParticleField& fieldDescriptor);

   public:
    explicit Particles(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, int ndims, std::vector<ParticleField> fields,
                       std::shared_ptr<particles::initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                       std::shared_ptr<mathFunctions::MathFunction> exactSolution);
    virtual ~Particles();

    const DM& GetParticleDM() const { return swarmDm; }
    PetscReal GetInitialTime() const { return timeInitial; }
    PetscReal GetFinalTime() const { return timeFinal; }
    TS GetTS() const { return particleTs; }

    /** Setup and size the subDomain with the subDomain **/
    virtual void Setup() override;

    /** Finalize the Setup of the subDomain before running **/
    virtual void Initialize() override;

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
