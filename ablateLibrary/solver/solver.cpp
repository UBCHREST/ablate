#include "solver.hpp"
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>
#include <regex>

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

void ablate::solver::Solver::DecompressFieldFieldDescriptor(std::vector<ablate::domain::FieldDescriptor>& fieldDescriptors) {

        for(auto& field : fieldDescriptors){
            for(std::size_t c = 0; c < field.components.size(); c++){
                if(field.components[c].find(domain::FieldDescriptor::DIMENSION) != std::string::npos){
                    auto baseName = field.components[c];

                    // Delete this component
                    field.components.erase(field.components.begin() + c);

                    for(PetscInt d =  subDomain->GetDimensions()-1; d >= 0; d--){
                        auto newName = std::regex_replace(baseName, std::regex(domain::FieldDescriptor::DIMENSION), std::to_string(d)); // replace 'def' -> 'klm'
                        field.components.insert(field.components.begin() + c, newName);
                    }
                }
            }
        }


}
void ablate::solver::Solver::PreStage(TS ts, PetscReal stagetime) {
    for(auto& function : preStageFunctions){
        function(ts, *this, stagetime);
    }
}
void ablate::solver::Solver::PreStep(TS ts) {
    for(auto& function : preStepFunctions){
        function(ts, *this);
    }
}
void ablate::solver::Solver::PostStep(TS ts) {
    for(auto& function : postStepFunctions){
        function(ts, *this);
    }
}
void ablate::solver::Solver::PostEvaluate(TS ts) {
    for(auto& function : postEvaluateFunctions){
        function(ts, *this);
    }
}
