#ifndef ABLATELIBRARY_COUPLEDPARTICLESOLVER_HPP
#define ABLATELIBRARY_COUPLEDPARTICLESOLVER_HPP

#include "particleSolver.hpp"
#include "processes/coupledProcess.hpp"
#include "solver/rhsFunction.hpp"

namespace ablate::particles {

/**
 * This is an extension of the particle solver that allows for fully coupled simulations.
 * The class implements the RHSFunction to allow inserting source terms back to main TS/flowfield
 */
class CoupledParticleSolver : public ParticleSolver, public ablate::solver::RHSFunction {
    // A string to hold the previous source term name name
    inline static const char PreviousPackedSolution[] = "PreviousPackedSolution";

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
     * @param coupledFields the fields to couple to the flow solver.  If not specified all solution fields will be coupled
     */
    CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<FieldDescription> fields,
                          std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                          std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {},
                          const std::vector<std::string>& coupledFields = {});

    /**
     * shared pointer version of the constructor
     * @param solverId
     * @param options
     * @param fields
     * @param processes
     * @param initializer
     * @param fieldInitialization
     * @param exactSolutions
     * @param coupledFields the fields to couple to the flow solver.  If not specified all solution fields will be coupled
     */
    CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, const std::vector<std::shared_ptr<FieldDescription>>& fields,
                          std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                          std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {},
                          const std::vector<std::string>& = {});

    //! cleanup any petsc objects
    ~CoupledParticleSolver() override;

    /** Override the Setup call in the subdomain to allow creating a new aux field that matches solution field in the domain**/
    void Setup() override;

    /** Override the Initialize call to set up localEulerianSourceVec**/
    void Initialize() override;

    /**
     * Map the source terms into the flow field once per time step (They are constant during the time step)
     * @param time
     * @param locX
     * @return
     */
    PetscErrorCode PreRHSFunction(TS ts, PetscReal time, bool initialStage, Vec locX) override;

    /**
     * Called to compute the RHS source term for the flow/macro TS
     * @param time
     * @param locX The locX vector includes boundary conditions
     * @param F
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locX, Vec locF) override;

   protected:
    /**
     * Override the macroStep for particles to enable coupling with the eulerian fields
     */
    void MacroStepParticles(TS macroTS, bool swarmMigrate) override;

   private:
    //! the processes that add source terms to the particle and domain ts
    std::vector<std::shared_ptr<processes::CoupledProcess>> coupledProcesses;

    //! the name of all fields to be coupled
    std::vector<std::string> coupledFieldsNames;

    //! the name of source terms in the particle to be coupled
    std::vector<std::string> coupledParticleFieldsNames;

    //! store a list of coupled fields
    std::vector<ablate::domain::Field> coupledFields;

    //! store the local vector for the cellDM source terms.  This is constant during a time step
    Vec localEulerianSourceVec{};
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_COUPLEDPARTICLESOLVER_HPP
