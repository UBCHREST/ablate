#include "solver.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::solver::Solver::Solver(std::string name,  std::shared_ptr<parameters::Parameters> options): name(name), petscOptions(nullptr) {
    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }

}

ablate::solver::Solver::~Solver(){
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(name, &petscOptions);
    }
}

void ablate::solver::Solver::SetupDomain(std::shared_ptr<ablate::domain::SubDomain> subDomainIn) {
    subDomain = subDomainIn;
}
