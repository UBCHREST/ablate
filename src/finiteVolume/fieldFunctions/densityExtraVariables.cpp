#include "densityExtraVariables.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/flowProcess.hpp"
#include "mathFunctions/functionPointer.hpp"

ablate::finiteVolume::fieldFunctions::DensityExtraVariables::DensityExtraVariables(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowStateIn,
                                                                                   std::vector<std::shared_ptr<mathFunctions::MathFunction>> evFunctions,
                                                                                   std::shared_ptr<ablate::domain::Region> region, std::string nonConservedFieldName)
    : ablate::mathFunctions::FieldFunction(nonConservedFieldName.empty() ? CompressibleFlowFields::DENSITY_EV_FIELD : CompressibleFlowFields::CONSERVED + nonConservedFieldName,
                                           std::make_shared<ablate::mathFunctions::FunctionPointer>(ComputeDensityEvFunction, this), {}, region),
      eulerFunction(flowStateIn->GetFieldFunction(CompressibleFlowFields::EULER_FIELD)),
      evFunctions(evFunctions) {}

PetscErrorCode ablate::finiteVolume::fieldFunctions::DensityExtraVariables::ComputeDensityEvFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto densityExtraVariables = (ablate::finiteVolume::fieldFunctions::DensityExtraVariables *)ctx;

    // compute the density at this location
    std::vector<PetscReal> eulerField(dim + 2, 0.0);
    densityExtraVariables->eulerFunction->GetPetscFunction()(dim, time, x, Nf, eulerField.data(), densityExtraVariables->eulerFunction->GetContext());

    // compute the mass fraction at this location
    try {
        for (PetscInt s = 0; s < PetscMin(Nf, (PetscInt)densityExtraVariables->evFunctions.size()); s++) {
            u[s] = eulerField[ablate::finiteVolume::CompressibleFlowFields::RHO] * densityExtraVariables->evFunctions[s]->Eval(x, dim, time);
        }
    } catch (std::exception &exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::FieldFunction, ablate::finiteVolume::fieldFunctions::DensityExtraVariables,
         "initializes the densityEV conserved field variables based upon a CompressibleFlowState and specified EV",
         ARG(ablate::finiteVolume::fieldFunctions::CompressibleFlowState, "state", "The CompressibleFlowState used to initialize"),
         ARG(std::vector<ablate::mathFunctions::MathFunction>, "functions", "The EV values in order"), OPT(ablate::domain::Region, "region", "A subset of the domain to apply the field function"),
         OPT(std::string, "name", "An optional name for the non-conserved field (default is CompressibleFlowFields::_EV_FIELD)"));
