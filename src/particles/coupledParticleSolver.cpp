#include "coupledParticleSolver.hpp"

#include <utility>
#include "utilities/vectorUtilities.hpp"

ablate::particles::CoupledParticleSolver::CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                std::vector<FieldDescription> fields, std::vector<std::shared_ptr<processes::Process>> processes,
                                                                std::shared_ptr<initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                                                                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : ParticleSolver(std::move(solverId), std::move(region), std::move(options), std::move(fields), std::move(processes), std::move(initializer), std::move(fieldInitialization),
                     std::move(exactSolutions)) {}

ablate::particles::CoupledParticleSolver::CoupledParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                                const std::vector<std::shared_ptr<FieldDescription>>& fields, std::vector<std::shared_ptr<processes::Process>> processes,
                                                                std::shared_ptr<initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                                                                std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : CoupledParticleSolver(std::move(solverId), std::move(region), std::move(options), ablate::utilities::VectorUtilities::Copy(fields), std::move(processes), std::move(initializer),
                            std::move(fieldInitialization), std::move(exactSolutions)) {}

ablate::particles::CoupledParticleSolver::~CoupledParticleSolver() {
    if (globalSourceTerms) {
        VecDestroy(&globalSourceTerms) >> utilities::PetscUtilities::checkError;
    }
}

PetscErrorCode ablate::particles::CoupledParticleSolver::ComputeRHSFunction(PetscReal time, Vec locX, Vec locF) {
    PetscFunctionBeginUser;

    // project the source terms to the global array
    const char* fieldnames[1] = {CoupledSourceTerm};
    Vec fields[1] = {globalSourceTerms};
    PetscCall(DMSwarmProjectFields(swarmDm, 1, fieldnames, fields, SCATTER_FORWARD));

    // Add them to the source loc f
    DMGlobalToLocalBegin(subDomain->GetDM(), locF, ADD_VALUES, locX);
    DMGlobalToLocalEnd(subDomain->GetDM(), locF, ADD_VALUES, locX);

    //Reset the source terms in the

    PetscFunctionReturn(PETSC_SUCCESS);
}

void ablate::particles::CoupledParticleSolver::Setup() {
    // Call the main particle setup
    ParticleSolver::Setup();

    // Build the source components from the subdomain
    std::vector<std::string> sourceComponents;

    for (const auto& field : subDomain->GetFields()) {
        if (field.numberComponents == 1) {
            sourceComponents.push_back(field.name);
        } else {
            for (const auto& component : field.components) {
                sourceComponents.push_back(field.name + "_" + component);
            }
        }
    }

    // Register a new aux field for the source terms to be passed to the main TS
    auto sourceField = FieldDescription{CoupledSourceTerm, domain::FieldLocation::AUX, sourceComponents};
    RegisterParticleField(sourceField);
}
void ablate::particles::CoupledParticleSolver::Initialize() {
    // Call the main particle Initialize
    ParticleSolver::Initialize();

    // Create a global vector for the mapping
    if (globalSourceTerms) {
        VecDestroy(&globalSourceTerms) >> utilities::PetscUtilities::checkError;
    }
    DMCreateGlobalVector(subDomain->GetDM(), &globalSourceTerms) >> utilities::PetscUtilities::checkError;
    VecZeroEntries(globalSourceTerms);
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::particles::CoupledParticleSolver, "Coupled Lagrangian particle solver", ARG(std::string, "id", "the name of the particle solver"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         OPT(std::vector<ablate::particles::FieldDescription>, "fields", "any additional fields beside coordinates"),
         ARG(std::vector<ablate::particles::processes::Process>, "processes", "the processes used to describe the particle source terms"),
         ARG(ablate::particles::initializers::Initializer, "initializer", "the initial particle setup methods"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "fieldInitialization", "the initial particle fields values"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "exactSolutions", "particle fields (SOL) exact solutions"));