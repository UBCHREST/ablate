#include "particles.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::particles::Particles::Particles(std::map<std::string, std::string> arguments, std::shared_ptr<particles::initializers::Initializer> initializer) : initializer(initializer) {
    utilities::PetscOptions::Set("particle_", arguments);
}

void ablate::particles::Particles::SetExactSolution(std::shared_ptr<mathFunctions::MathFunction> exactSolutionIn) {
    exactSolution = exactSolutionIn;

    ParticleSetExactSolutionFlow(particleData, exactSolution->GetPetscFunction(), exactSolution->GetContext()) >> checkError;
}
void ablate::particles::Particles::InitializeFlow(std::shared_ptr<flow::Flow> flow, std::shared_ptr<solve::TimeStepper> flowTimeStepper) {
    // link the flow to the particles
    ParticleInitializeFlow(particleData, flow->GetMesh().GetDomain(), flow->GetFlowSolution()) >> checkError;

    // name the particle domain
    PetscObjectSetOptionsPrefix((PetscObject)(particleData->dm), "particle_") >> checkError;
    PetscObjectSetName((PetscObject)particleData->dm, "Particles") >> checkError;

    // initialize the particles
    initializer->Initialize(*flow, particleData->dm);
    DMViewFromOptions(particleData->dm, NULL, "-dm_view") >> checkError;

    // Setup particle position integrator
    TSCreate(PETSC_COMM_WORLD, &particleTs) >> checkError;
    PetscObjectSetOptionsPrefix((PetscObject)particleTs, "particle_") >> checkError;
}
