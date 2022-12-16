#include "petscUtilities.hpp"
#include "environment/runEnvironment.hpp"
#include "petscError.hpp"

void ablate::utilities::PetscUtilities::Initialize(const char help[]) {
    PetscInitialize(ablate::environment::RunEnvironment::GetArgCount(), ablate::environment::RunEnvironment::GetArgs(), nullptr, help) >> checkError;

    // register the cleanup
    ablate::environment::RunEnvironment::RegisterCleanUpFunction("ablate::utilities::PetscUtilities::Initialize", []() { PetscFinalize() >> checkError; });
}
