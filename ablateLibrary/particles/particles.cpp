#include "particles.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::particles::Particles::Particles(std::string name, std::map<std::string, std::string> arguments, std::shared_ptr<particles::initializers::Initializer> initializer) : name(name), initializer(initializer) {
    utilities::PetscOptions::Set(name + "_", arguments);
}

void ablate::particles::Particles::SetExactSolution(std::shared_ptr<mathFunctions::MathFunction> exactSolutionIn) {
    exactSolution = exactSolutionIn;

    ParticleSetExactSolutionFlow(particleData, exactSolution->GetPetscFunction(), exactSolution->GetContext()) >> checkError;
}
void ablate::particles::Particles::InitializeFlow(std::shared_ptr<flow::Flow> flow, std::shared_ptr<solve::TimeStepper> flowTimeStepper) {
    // link the flow to the particles
    ParticleInitializeFlow(particleData, flow->GetFlowData()) >> checkError;

    // name the particle domain
    auto namePrefix = name + "_";
    PetscObjectSetOptionsPrefix((PetscObject)(particleData->dm), namePrefix.c_str()) >> checkError;
    PetscObjectSetName((PetscObject)particleData->dm, name.c_str()) >> checkError;

    // initialize the particles
    initializer->Initialize(*flow, particleData->dm);

    // Setup particle position integrator
    TSCreate(PETSC_COMM_WORLD, &particleTs) >> checkError;
    PetscObjectSetOptionsPrefix((PetscObject)particleTs, namePrefix.c_str()) >> checkError;
}
