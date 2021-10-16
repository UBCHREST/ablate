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


void ablate::solver::Solver::Save(PetscViewer viewer, PetscInt steps, PetscReal time) const {
    auto dm = subDomain->GetDM();
    auto auxDM = subDomain->GetAuxDM();
    // If this is the first output, save the mesh
    if (steps == 0) {
        // Print the initial mesh
        DMView(dm, viewer) >> checkError;
    }

    // set the dm sequence number, because we may be skipping outputs
    DMSetOutputSequenceNumber(dm, steps, time) >> checkError;
    if (auxDM) {
        DMSetOutputSequenceNumber(auxDM, steps, time) >> checkError;
    }

    // Always save the main flowField
    VecView(subDomain->GetSolutionVector(), viewer) >> checkError;

    // If there is aux data output
    if (subDomain->GetAuxVector()) {
        // copy over the sequence data from the main dm
        PetscReal dmTime;
        PetscInt dmSequence;
        DMGetOutputSequenceNumber(dm, &dmSequence, &dmTime) >> checkError;
        DMSetOutputSequenceNumber(auxDM, dmSequence, dmTime) >> checkError;

        Vec auxGlobalField;
        DMGetGlobalVector(auxDM, &auxGlobalField) >> checkError;

        // copy over the name of the auxFieldVector
        const char* tempName;
        PetscObjectGetName((PetscObject)subDomain->GetAuxVector(), &tempName) >> checkError;
        PetscObjectSetName((PetscObject)auxGlobalField, tempName) >> checkError;
        DMLocalToGlobal(auxDM, subDomain->GetAuxVector(), INSERT_VALUES, auxGlobalField) >> checkError;
        VecView(auxGlobalField, viewer) >> checkError;
        DMRestoreGlobalVector(auxDM, &auxGlobalField) >> checkError;
    }

//    if (!ex.empty()) {
//        Vec exactVec;
//        DMGetGlobalVector(dm->GetDomain(), &exactVec) >> checkError;
//
//        // Get the number of fields
//        PetscDS ds;
//        DMGetDS(dm->GetDomain(), &ds) >> checkError;
//        PetscInt numberOfFields;
//        PetscDSGetNumFields(ds, &numberOfFields) >> checkError;
//        std::vector<ablate::mathFunctions::PetscFunction> exactFuncs(numberOfFields);
//        std::vector<void*> exactCtxs(numberOfFields);
//        for (auto f = 0; f < numberOfFields; ++f) {
//            PetscDSGetExactSolution(ds, f, &exactFuncs[f], &exactCtxs[f]) >> checkError;
//            if (!exactFuncs[f]) {
//                throw std::invalid_argument("The exact solution has not set");
//            }
//        }
//
//        DMProjectFunction(dm->GetDomain(), time, &exactFuncs[0], &exactCtxs[0], INSERT_ALL_VALUES, exactVec) >> checkError;
//
//        PetscObjectSetName((PetscObject)exactVec, "exact") >> checkError;
//        VecView(exactVec, viewer) >> checkError;
//        DMRestoreGlobalVector(dm->GetDomain(), &exactVec) >> checkError;
//    }
}

void ablate::solver::Solver::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // The only item that needs to be explicitly restored is the flowField
    DMSetOutputSequenceNumber(subDomain->GetDM(), sequenceNumber, time) >> checkError;
    VecLoad(subDomain->GetSolutionVector(), viewer) >> checkError;
}
