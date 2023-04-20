#include "chemTabField.hpp"
#include <algorithm>
#include <mathFunctions/functionPointer.hpp>
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::fieldFunctions::ChemTabField::ChemTabField(const std::string &initializer, std::shared_ptr<ablate::eos::EOS> eos)
    : ablate::mathFunctions::FieldFunction(eos::EOS::PROGRESS,
                                           std::make_shared<ablate::mathFunctions::FunctionPointer>(ablate::finiteVolume::fieldFunctions::ChemTabField::ComputeChemTabProgress, this)) {
    auto chemTabModel = std::dynamic_pointer_cast<ablate::eos::ChemTab>(eos);
    if (!chemTabModel) {
        throw std::invalid_argument("The ablate::finiteVolume::fieldFunctions::ChemTabField requires a ablate::eos::ChemTab model");
    }

    // Get the progress variables from ChemTab
    progressVariables.resize(chemTabModel->GetProgressVariables().size());
    chemTabModel->GetInitializerProgressVariables(initializer, progressVariables);
}

PetscErrorCode ablate::finiteVolume::fieldFunctions::ChemTabField::ComputeChemTabProgress(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *progress, void *ctx) {
    PetscFunctionBeginUser;
    auto chemTabField = (ablate::finiteVolume::fieldFunctions::ChemTabField *)ctx;

    // Sanity Check
    if (Nf != (PetscInt)chemTabField->progressVariables.size()) {
        SETERRQ(PETSC_COMM_SELF,
                PETSC_ERR_LIB,
                "There is a miss match in the number of ComputeChemTabProgress variables %" PetscInt_FMT " vs %" PetscInt_FMT,
                Nf,
                (PetscInt)chemTabField->progressVariables.size());
    }

    for (std::size_t i = 0; i < chemTabField->progressVariables.size(); i++) {
        progress[i] = chemTabField->progressVariables[i];
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::FieldFunction, ablate::finiteVolume::fieldFunctions::ChemTabField, "Class that species progress variable from a chemTab model",
         ARG(std::string, "initializer", "the name of the initializer in the chemTab model"), ARG(ablate::eos::EOS, "eos", "must be a ablate::eos::ChemTab model"));