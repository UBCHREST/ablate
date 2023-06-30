#ifndef ABLATELIBRARY_TIMESTEPPER_H
#define ABLATELIBRARY_TIMESTEPPER_H
#include <petsc.h>
#include <functional>
#include <io/serializer.hpp>
#include <map>
#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "boundaryFunction.hpp"
#include "domain/domain.hpp"
#include "domain/initializer.hpp"
#include "iFunction.hpp"
#include "monitors/monitor.hpp"
#include "physicsTimeStepFunction.hpp"
#include "rhsFunction.hpp"
#include "solver.hpp"
#include "utilities/loggable.hpp"
#include "utilities/staticInitializer.hpp"

namespace ablate::solver {
class TimeStepper : public std::enable_shared_from_this<TimeStepper>, private utilities::Loggable<TimeStepper>, private utilities::StaticInitializer {
   public:
    /**
     * Optional initializer that can be called for TS before the first time step (TSSolve)
     */
    using AdaptInitializer = std::function<void(TS ts, TSAdapt adapt)>;

   private:
    TS ts{};                                                                         /** The PETSC time stepper**/
    std::string name;                                                                /** the name for this time stepper **/
    std::map<std::string, std::vector<std::shared_ptr<monitors::Monitor>>> monitors; /** the monitors **/

    // Hold a const value of the domain
    const std::shared_ptr<ablate::domain::Domain> domain;

    // Store a pointer to the Serializer
    const std::shared_ptr<io::Serializer> serializer;

    // Hold a list of solvers
    std::vector<std::shared_ptr<ablate::solver::Solver>> solvers;

    // hold a bool to determine if this domain has been initialized
    bool initialized = false;

    // Keep track of the current stage count, this is a hack to know if we are in a preStep/preStage for rhs
    bool runInitialStep = false;

    // If true, uses a slow nan/inf check at each source term for each evaluation
    const bool verboseSourceCheck;

    /**
     * The TSPre*Function is used to call both th PreStep (once) and PreStage (as need calls).
     *
     * @param ts
     * @param stagetime
     * @return
     */
    ///@{
    static PetscErrorCode TSPreStageFunction(TS ts, PetscReal stagetime);
    static PetscErrorCode TSPreStepFunction(TS ts);
    static PetscErrorCode TSPostStepFunction(TS ts);
    static PetscErrorCode TSPostEvaluateFunction(TS ts);
    ///@}
    // store a list of functions for each evaluation type
    std::vector<std::shared_ptr<IFunction>> iFunctionSolvers;
    std::vector<std::shared_ptr<RHSFunction>> rhsFunctionSolvers;
    std::vector<std::shared_ptr<BoundaryFunction>> boundaryFunctionSolvers;
    std::vector<std::shared_ptr<PhysicsTimeStepFunction>> physicsTimeStepFunctionSolvers;

    // support for function residual/jacobian evaluation
    static PetscErrorCode SolverComputeBoundaryFunctionLocal(DM dm, PetscReal time, Vec locX, Vec locX_t, void *timeStepperCtx);
    static PetscErrorCode SolverComputeIFunctionLocal(DM dm, PetscReal time, Vec locX, Vec locX_t, Vec locF, void *timeStepperCtx);
    static PetscErrorCode SolverComputeIJacobianLocal(DM dm, PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, void *domainCtx);

    /**
     * Note this is not a Local function.  It includes fixes for Petsc TSComputeRHSFunction_DMLocal
     * @return
     */
    static PetscErrorCode SolverComputeRHSFunction(TS ts, PetscReal time, Vec X, Vec F, void *ctx);

    // store the list of field initializations
    const std::shared_ptr<ablate::domain::Initializer> initializations;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> absoluteTolerances;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> relativeTolerances;

    //! hold a list of static AdaptInitializers
    static inline std::map<std::string, AdaptInitializer> adaptInitializers;

   public:
    /**
     * primary constructor for timestepper
     * @param name
     * @param domain
     * @param arguments
     * @param serializer
     * @param initialization
     * @param exactSolutions
     * @param absoluteTolerances
     * @param relativeTolerances
     * @param verboseSourceCheck
     */
    TimeStepper(const std::string &name, std::shared_ptr<ablate::domain::Domain> domain, const std::shared_ptr<ablate::parameters::Parameters> &arguments = {},
                std::shared_ptr<io::Serializer> serializer = {}, std::shared_ptr<ablate::domain::Initializer> initialization = {},
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {}, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> absoluteTolerances = {},
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> relativeTolerances = {}, bool verboseSourceCheck = {});

    /**
     * primary constructor for timestepper without an unqiue name
     * @param name
     * @param domain
     * @param arguments
     * @param serializer
     * @param initialization
     * @param exactSolutions
     * @param absoluteTolerances
     * @param relativeTolerances
     * @param verboseSourceCheck
     */
    TimeStepper(std::shared_ptr<ablate::domain::Domain> domain, const std::shared_ptr<ablate::parameters::Parameters> &arguments = {}, std::shared_ptr<io::Serializer> serializer = {},
                std::shared_ptr<ablate::domain::Initializer> initialization = {}, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions = {},
                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> absoluteTolerances = {}, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> relativeTolerances = {},
                bool verboseSourceCheck = {});

    ~TimeStepper();

    TS &GetTS() { return ts; }

    Vec GetSolutionVector() { return domain->GetSolutionVector(); }

    /**
     * Optional call to Initialize before setup
     */
    void Initialize();

    /**
     * Initializes if needed, then calls solve
     */
    void Solve();

    /**
     * Computes the physics based time step that can be used to control the adaptive time step
     * @return
     */
    PetscErrorCode ComputePhysicsTimeStep(PetscReal *dt);

    void Register(const std::shared_ptr<ablate::solver::Solver> &solver, const std::vector<std::shared_ptr<monitors::Monitor>> & = {});

    double GetTime() const;

    const std::string &GetName() const { return name; }

    /**
     * Function to register the adapt by name
     * @param adaptName This must be the same same used to create the class
     * @param adaptInitializer
     */
    static void RegisterAdaptInitializer(std::string adaptName, AdaptInitializer adaptInitializer) { adaptInitializers[adaptName] = adaptInitializer; }
};
}  // namespace ablate::solver

#endif  // ABLATELIBRARY_TIMESTEPPER_H
