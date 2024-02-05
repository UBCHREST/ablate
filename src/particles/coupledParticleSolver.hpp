#ifndef ABLATELIBRARY_COUPLEDPARTICLESOLVER_HPP
#define ABLATELIBRARY_COUPLEDPARTICLESOLVER_HPP

#include "particleSolver.hpp"
#include "solver/rhsFunction.hpp"

namespace ablate::particles {

/**
 * This is an extension of the particle solver that allows for fully coupled simulations.
 * The class implements the RHSFunction to allow inserting source terms back to main TS/flowfield
 */
class CoupledParticleSolver : public ParticleSolver, public ablate::solver::RHSFunction {
   public:
    // A string to hold the coupled source terms name
    inline static const char CoupledSourceTerm[] = "CoupledSourceTerm";

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
    CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, std::vector<FieldDescription> fields,
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
    CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region>, std::shared_ptr<parameters::Parameters> options, const std::vector<std::shared_ptr<FieldDescription>>& fields,
                          std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                          std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {});

    /**
     * Allow cleanup of the scratch vec
     */
    ~CoupledParticleSolver() override;

    /** Override the Setup call in the subdomain to allow creating a new aux field that matches solution field in the domain**/
    void Setup() override;

    /*** Set up mesh dependent initialization, this may be called multiple times if the mesh changes **/
    void Initialize() override;

    /**
     * Called to compute the RHS source term.
     * @param time
     * @param locX The locX vector includes boundary conditions
     * @param F
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locX, Vec locF) override;

   private:
    //! store a temporary global vector for the source terms, this is required for the projection
    Vec globalSourceTerms = {};
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_COUPLEDPARTICLESOLVER_HPP
