#ifndef ABLATELIBRARY_PARTICLESOLVER_HPP
#define ABLATELIBRARY_PARTICLESOLVER_HPP

#include "particleField.hpp"
#include "particles/initializers/initializer.hpp"
#include "processes/process.hpp"
#include "solver/solver.hpp"

namespace ablate::particles {

class ParticleSolver : public solver::Solver {
   public:
    /** common field names for particles **/
    inline static const char ParticleVelocity[] = "ParticleVelocity";
    inline static const char ParticleDiameter[] = "ParticleDiameter";
    inline static const char ParticleDensity[] = "ParticleDensity";
    inline static const char PackedSolution[] = "PackedSolution";
    inline static const char ParticleInitialLocation[] = "InitialLocation";

   private:
    //!  particle dm, this is a swarm
    DM swarmDm = nullptr;

    //! time integration data
    TS particleTs = nullptr;

    //! The time for ui, at the beginning of the advection solve
    PetscReal timeInitial = 0.0;

    //! The time for uf, at the end of the advection solve
    PetscReal timeFinal = 0.0;

    //! the dims from the subdomain
    PetscInt ndims = 0;

    //! store a boolean to state if a dmChanged (number of particles local/global changed)
    bool dmChanged = false;

    //! the fields to create in the particle solver
    std::vector<ParticleField> fields;

    //! fields that interact with the particle ts
    std::vector<ParticleField> solutionFields;

    //! the processes that add source terms to the particle and domain ts
    std::vector<std::shared_ptr<processes::Process>> processes;

    //! Store the particle location and field initialization
    std::shared_ptr<particles::initializers::Initializer> initializer = nullptr;

    //! initialize other particle variables
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization;

    //! store the exact solution if provided
    const std::shared_ptr<mathFunctions::MathFunction> exactSolution = nullptr;

   public:
    ParticleSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<ParticleField> fields,
                   std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                   std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::shared_ptr<mathFunctions::MathFunction> exactSolution);

    ~ParticleSolver() override;

    /** Setup and size the subDomain with the subDomain **/
    void Setup() override;

    /*** Set up mesh dependent initialization, this may be called multiple times if the mesh changes **/
    void Initialize() override;

   private:
    /**
     * The register fields adds the field to the swarm
     * @param fieldDescriptor
     */
    void RegisterParticleField(const ParticleField& fieldDescriptor);

    /**
     * stores the initial particle locations in the ParticleInitialLocation field
     */
    void StoreInitialParticleLocations();

    /**
     * computes the error based upon the the specified exact solution and initial particle location
     * @param particleTS
     * @param u
     * @param e
     * @return
     */
    static PetscErrorCode ComputeParticleError(TS particleTS, Vec u, Vec e);

    /**
     * Project the field function to the particle field
     */
    void ProjectFunction(const std::shared_ptr<mathFunctions::FieldFunction>& fieldFunction);
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_PARTICLESOLVER_HPP
