#include "solver.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::flow::Solver::Solver(std::string name,  std::shared_ptr<parameters::Parameters> options): name(name), petscOptions(nullptr) {
    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }

}

ablate::flow::Solver::~Solver(){
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(name, &petscOptions);
    }
}

void ablate::flow::Solver::SetupDomain(std::shared_ptr<ablate::domain::SubDomain> subDomainIn) {
    subDomain = subDomainIn;
}
