#ifndef ABLATELIBRARY_PARTICLESOLVER_HPP
#define ABLATELIBRARY_PARTICLESOLVER_HPP

#include "field.hpp"
#include "fieldDescription.hpp"
#include "initializers/initializer.hpp"
#include "processes/process.hpp"
#include "solver/solver.hpp"

namespace ablate::particles {

class ParticleSolver : public solver::Solver, public io::Serializable {
   public:
    /** common field names for particles **/
    inline static const char ParticleVelocity[] = "ParticleVelocity";
    inline static const char ParticleDiameter[] = "ParticleDiameter";
    inline static const char ParticleDensity[] = "ParticleDensity";
    inline static const char PackedSolution[] = "PackedSolution";
    inline static const char ParticleInitialLocation[] = "InitialLocation";

    //! These coordinates are part of the solution vector
    inline static const char ParticleCoordinates[] = "coordinates";

   protected:
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

    //! the fields specific to be created to create in the particle solver
    std::vector<FieldDescription> fieldsDescriptions;

    //! all fields in the particle solver
    std::vector<Field> fields;

    //! a map of fields for easy field lookup
    std::map<std::string, Field> fieldsMap;

    //! the processes that add source terms to the particle and domain ts
    std::vector<std::shared_ptr<processes::Process>> processes;

    //! Store the particle location and field initialization
    std::shared_ptr<particles::initializers::Initializer> initializer = nullptr;

    //! initialize other particle variables
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization;

    //! store the exact solution if provided
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;

   public:
    /**
     * default constructor
     * @param solverId
     * @param options
     * @param fields
     * @param processes
     * @param initializer
     * @param fieldInitialization
     * @param exactSolutions
     */
    ParticleSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<FieldDescription> fields,
                   std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                   std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});

    /**
     * shared pointer version of the constructor
     * @param solverId
     * @param options
     * @param fields
     * @param processes
     * @param initializer
     * @param fieldInitialization
     * @param exactSolutions
     */
    ParticleSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, const std::vector<std::shared_ptr<FieldDescription>>& fields,
                   std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                   std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});

    ~ParticleSolver() override;

    /** Setup and size the subDomain with the subDomain **/
    void Setup() override;

    /*** Set up mesh dependent initialization, this may be called multiple times if the mesh changes **/
    void Initialize() override;

    /**
     * Function to be be called after each flow time step
     */
    void MacroStepParticles(TS macroTS);

    /**
     * return access to the particle dm
     * @return the swamParticle dm
     */
    inline DM GetParticleDM() { return swarmDm; }

    /**
     * return access to the particle dm
     * @return the swamParticle dm
     */
    inline TS GetParticleTS() { return particleTs; }

    /**
     * Helper function useful for tests
     * @param particleTS
     * @return
     */
    static PetscErrorCode ComputeParticleExactSolution(TS particleTS, Vec);

    /**
     * only required function, returns the id of the object.  Should be unique for the simulation
     * @return
     */
    [[nodiscard]] const std::string& GetId() const override { return GetSolverId(); }

    /**
     * shared function to view all particles;
     * @param viewer
     * @param steps
     * @param time
     * @param u
     */
    PetscErrorCode Save(PetscViewer viewer, PetscInt steps, PetscReal time) override;

    /**
     * shared function to view all particles;
     * @param viewer
     * @param steps
     * @param time
     * @param u
     */
    PetscErrorCode Restore(PetscViewer viewer, PetscInt steps, PetscReal time) override;

   protected:
    /**
     * The register fields adds the field to the swarm
     * @param fieldDescriptor
     */
    void RegisterParticleField(const FieldDescription& fieldDescriptor);

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
    void ProjectFunction(const std::shared_ptr<mathFunctions::FieldFunction>& fieldFunction, PetscReal time = 0.0);

    /**
     * Migrate the particle between ranks based upon the background mesh
     */
    void SwarmMigrate();

    /**
     * map the coordinates to the solution vector
     */
    void CoordinatesToSolutionVector();

    /**
     * map the solution vector coordinates to the particle coordinates
     */
    void CoordinatesFromSolutionVector();

   protected:
    /**
     * Get the array based upon field
     */
    template <class T>
    void GetField(const Field& field, T** values) {
        if (field.location == domain::FieldLocation::SOL) {
            // Get the solution vector
            DMSwarmGetField(swarmDm, PackedSolution, nullptr, nullptr, (void**)values) >> utilities::PetscUtilities::checkError;
        } else {
            // get the raw field
            DMSwarmGetField(swarmDm, field.name.c_str(), nullptr, nullptr, (void**)values) >> utilities::PetscUtilities::checkError;
        }
    }

    /**
     * Restore the array and field information based upon field name
     */
    template <class T>
    void RestoreField(const Field& field, T** values) {
        if (field.location == domain::FieldLocation::SOL) {
            // Get the solution vector
            DMSwarmRestoreField(swarmDm, PackedSolution, nullptr, nullptr, (void**)values) >> utilities::PetscUtilities::checkError;
        } else {
            // get the raw field
            DMSwarmRestoreField(swarmDm, field.name.c_str(), nullptr, nullptr, (void**)values) >> utilities::PetscUtilities::checkError;
        }
    }

    /**
     * Get the array and field information based upon field name
     */
    template <class T>
    const Field& GetField(const std::string& fieldName, T** values) {
        const auto& field = GetField(fieldName);
        GetField(field, values);
        return field;
    }

    /**
     * Restore the array and field information based upon field name
     */
    template <class T>
    void RestoreField(const std::string& fieldName, T** values) {
        const auto& field = GetField(fieldName);
        RestoreField(field, values);
    }

    /**
     * Get field information based upon field name
     */
    [[nodiscard]] const Field& GetField(const std::string& fieldName) const { return fieldsMap.at(fieldName); }

    /**
     * computes the particle rhs for the particle TS
     * @param ts
     * @param t
     * @param X
     * @param F
     * @param ctx
     * @return
     */
    static PetscErrorCode ComputeParticleRHS(TS ts, PetscReal t, Vec X, Vec F, void* ctx);
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_PARTICLESOLVER_HPP
