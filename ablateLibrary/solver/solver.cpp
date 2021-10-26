#include "solver.hpp"
#include <regex>
#include <utilities/petscError.hpp>
#include <utilities/petscOptions.hpp>

ablate::solver::Solver::Solver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options)
    : solverId(solverId), region(region), petscOptions(nullptr) {
    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }
}

ablate::solver::Solver::~Solver() {
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(GetId(), &petscOptions);
    }
}

void ablate::solver::Solver::Register(std::shared_ptr<ablate::domain::SubDomain> subDomainIn) { subDomain = subDomainIn; }

void ablate::solver::Solver::DecompressFieldFieldDescriptor(std::vector<ablate::domain::FieldDescriptor>& fieldDescriptors) {
    for (auto& field : fieldDescriptors) {
        for (std::size_t c = 0; c < field.components.size(); c++) {
            if (field.components[c].find(domain::FieldDescriptor::DIMENSION) != std::string::npos) {
                auto baseName = field.components[c];

                // Delete this component
                field.components.erase(field.components.begin() + c);

                for (PetscInt d = subDomain->GetDimensions() - 1; d >= 0; d--) {
                    auto newName = std::regex_replace(baseName, std::regex(domain::FieldDescriptor::DIMENSION), std::to_string(d));  // replace 'def' -> 'klm'
                    field.components.insert(field.components.begin() + c, newName);
                }
            }
        }
    }
}
void ablate::solver::Solver::PreStage(TS ts, PetscReal stagetime) {
    for (auto& function : preStageFunctions) {
        function(ts, *this, stagetime);
    }
}
void ablate::solver::Solver::PreStep(TS ts) {
    for (auto& function : preStepFunctions) {
        function(ts, *this);
    }
}
void ablate::solver::Solver::PostStep(TS ts) {
    for (auto& function : postStepFunctions) {
        function(ts, *this);
    }
}
void ablate::solver::Solver::PostEvaluate(TS ts) {
    for (auto& function : postEvaluateFunctions) {
        function(ts, *this);
    }
}

void ablate::solver::Solver::Save(PetscViewer viewer, PetscInt steps, PetscReal time) const {
    auto subDm = subDomain->GetSubDM();
    auto auxDM = subDomain->GetSubAuxDM();
    // If this is the first output, save the mesh
    if (steps == 0) {
        // Print the initial mesh
        DMView(subDm, viewer) >> checkError;
    }

    // set the dm sequence number, because we may be skipping outputs
    DMSetOutputSequenceNumber(subDm, steps, time) >> checkError;
    if (auxDM) {
        DMSetOutputSequenceNumber(auxDM, steps, time) >> checkError;
    }

    // Always save the main flowField
    VecView(subDomain->GetSubSolutionVector(), viewer) >> checkError;

    // If there is aux data output
    if (auto subAuxVector = subDomain->GetSubAuxVector()) {
        // copy over the sequence data from the main dm
        PetscReal dmTime;
        PetscInt dmSequence;
        DMGetOutputSequenceNumber(subDm, &dmSequence, &dmTime) >> checkError;
        DMSetOutputSequenceNumber(auxDM, dmSequence, dmTime) >> checkError;

        Vec auxGlobalField;
        DMGetGlobalVector(auxDM, &auxGlobalField) >> checkError;

        // copy over the name of the auxFieldVector
        const char* tempName;
        PetscObjectGetName((PetscObject)subAuxVector, &tempName) >> checkError;
        PetscObjectSetName((PetscObject)auxGlobalField, tempName) >> checkError;
        DMLocalToGlobal(auxDM, subAuxVector, INSERT_VALUES, auxGlobalField) >> checkError;
        VecView(auxGlobalField, viewer) >> checkError;
        DMRestoreGlobalVector(auxDM, &auxGlobalField) >> checkError;
    }
}

void ablate::solver::Solver::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    // The only item that needs to be explicitly restored is the flowField
    DMSetOutputSequenceNumber(subDomain->GetDM(), sequenceNumber, time) >> checkError;
    VecLoad(subDomain->GetSolutionVector(), viewer) >> checkError;
}
