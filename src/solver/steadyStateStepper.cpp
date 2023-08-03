#include "steadyStateStepper.hpp"
#include <io/interval/fixedInterval.hpp>
#include <utility>
#include "criteria/convergenceException.hpp"

ablate::solver::SteadyStateStepper::SteadyStateStepper(std::shared_ptr<ablate::domain::Domain> domain, std::vector<std::shared_ptr<ablate::solver::criteria::ConvergenceCriteria>> convergenceCriteria,
                                                       const std::shared_ptr<ablate::parameters::Parameters>& arguments, std::shared_ptr<ablate::io::Serializer> serializer,
                                                       std::shared_ptr<ablate::domain::Initializer> initialization,
                                                       std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> absoluteTolerances,
                                                       std::vector<std::shared_ptr<ablate::mathFunctions::FieldFunction>> relativeTolerances, bool verboseSourceCheck,
                                                       std::shared_ptr<ablate::monitors::logs::Log> log, int checkIntervalIn)
    : ablate::solver::TimeStepper(std::move(domain), arguments, std::move(serializer), std::move(initialization), {} /* no exact solution for stead state solver */, std::move(absoluteTolerances),
                                  std::move(relativeTolerances), verboseSourceCheck),
      checkInterval(checkIntervalIn ? checkIntervalIn : 100),
      convergenceCriteria(std::move(convergenceCriteria)),
      log(std::move(log)) {}

ablate::solver::SteadyStateStepper::~SteadyStateStepper() = default;

bool ablate::solver::SteadyStateStepper::Initialize() {
    // Call the super class to determine if it needs
    bool justInitialized = TimeStepper::Initialize();

    // if this was justInitialized also, set up the criteria
    if (justInitialized) {
        for (auto& criterion : convergenceCriteria) {
            criterion->Initialize(GetDomain());
        }
    }

    return justInitialized;
}

void ablate::solver::SteadyStateStepper::Solve() {
    // Do the basic initialize
    Initialize();

    // Store the max steps to use for a final check
    TSGetMaxSteps(GetTS(), &maxSteps) >> ablate::utilities::PetscUtilities::checkError;

    // Get the current step and time.  For restart cases this step might not be zero
    PetscInt step;
    TSGetStepNumber(GetTS(), &step) >> ablate::utilities::PetscUtilities::checkError;

    // Set the initial max time steps
    TSSetMaxSteps(GetTS(), checkInterval + step) >> ablate::utilities::PetscUtilities::checkError;

    // perform a basic solve (this will set up anything else that needs to be setup)
    TimeStepper::Solve();

    // keep stepping until convergence is reached
    bool converged = false;

    // step until convergence is reached
    while (!converged) {
        // Solve to the next number of steps
        TSSolve(GetTS(), nullptr) >> ablate::utilities::PetscUtilities::checkError;

        // check for convergence.  Set converged to true before checking
        converged = true;

        // Get the current step and time
        TSGetStepNumber(GetTS(), &step) >> ablate::utilities::PetscUtilities::checkError;

        PetscReal time;
        TSGetTime(GetTS(), &time) >> ablate::utilities::PetscUtilities::checkError;

        // check each criterion
        if (log) {
            log->Printf("Checking convergence after %" PetscInt_FMT " steps.\n", step);
        }
        for (auto& criterion : convergenceCriteria) {
            converged = converged && criterion->CheckConvergence(GetDomain(), time, step, log);
        }

        // report log status
        if (log) {
            if (converged) {
                log->Printf("\tConvergence reached after %" PetscInt_FMT " steps.\n", step);
            }
            if (!converged) {
                log->Printf("\tSolution not converged after %" PetscInt_FMT " steps.\n", step);
            }
        }

        // check for max steps
        if (!converged && step > maxSteps) {
            if (log) {
                log->Printf("Failed to converge after %" PetscInt_FMT " steps.\n", step);
            }
            throw criteria::ConvergenceException("Failed to converge after " + std::to_string(step) + ".\n");
        }

        // Increase the number of steps and try again
        TSSetMaxSteps(GetTS(), checkInterval + step) >> ablate::utilities::PetscUtilities::checkError;
    }
}

#include "registrar.hpp"
REGISTER(ablate::solver::TimeStepper, ablate::solver::SteadyStateStepper, "a time stepper designed to march to steady state", ARG(ablate::domain::Domain, "domain", "the mesh used for the simulation"),
         OPT(std::vector<ablate::solver::criteria::ConvergenceCriteria>, "criteria", "the criteria used to determine when the solution is converged"),
         OPT(ablate::parameters::Parameters, "arguments", "arguments to be passed to petsc"), OPT(ablate::io::Serializer, "io", "the serializer used with this timestepper"),
         OPT(ablate::domain::Initializer, "initialization", "initialization field functions"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "absoluteTolerances", "optional absolute tolerances for a field"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "relativeTolerances", "optional relative tolerances for a field"),
         OPT(bool, "verboseSourceCheck", "does a slow nan/inf for solvers that use rhs evaluation. This is slow and should only be used for debug."),
         OPT(ablate::monitors::logs::Log, "log", "optionally log the convergence history"), OPT(int, "checkInterval", "the number of steps between criteria checks"));
