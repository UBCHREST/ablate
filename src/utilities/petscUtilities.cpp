#include "petscUtilities.hpp"
#include "environment/runEnvironment.hpp"
#include "petscError.hpp"

void ablate::utilities::PetscUtilities::Initialize(int* argc, char*** args, const char help[]) {
    PetscInitialize(argc, args, nullptr, help) >> checkError;

    // register the cleanup
    ablate::environment::RunEnvironment::RegisterCleanUpFunction("ablate::utilities::PetscUtilities::Initialize", []() { PetscFinalize() >> checkError; });
}
