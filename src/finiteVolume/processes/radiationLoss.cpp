#include "radiationLoss.hpp"

ablate::finiteVolume::processes::RadiationLoss::RadiationLoss(std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions) : functions(std::move(functions)) {}

void ablate::finiteVolume::processes::RadiationLoss::Setup(ablate::finiteVolume::FiniteVolumeSolver &fvmSolver) {
    for (const auto &[fieldName, function] : functions) {
        // Get the field from the subDomain
        const auto &field = fvmSolver.GetSubDomain().GetField(fieldName);

        petscFunctions.emplace_back(PetscFunctionStruct{.petscFunction = function->GetPetscFunction(), .petscContext = function->GetContext(), .fieldSize = field.numberComponents});

        // add the source function
        fvmSolver.RegisterRHSFunction(ComputeRadiationLoss, &petscFunctions.back(), {fieldName}, {}, {});
    }

    absorptivityFunction = GetRadiationModel()->GetRadiationPropertiesTemperatureFunction(eos::radiationProperties::RadiationProperty::Absorptivity, fvmSolver.GetSubDomain().GetFields());
}

PetscErrorCode ablate::finiteVolume::processes::RadiationLoss::ComputeRadiationLoss(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u,
                                                                                    const PetscInt *aOff, const PetscScalar *a, PetscScalar *f, void *ctx) {
    PetscFunctionBegin;
    auto absorptivityFunctionContext = absorptivityFunction.context.get();  //!< Get access to the absorption function
    double kappa = 1;                  //!< Absorptivity coefficient, property of each cell
    DMPlexPointLocalFieldRead(auxDm, iCell, temperatureFieldInfo.id, auxArray, &temperature) >> utilities::PetscUtilities::checkError;

    /**
     * Get the rhs values so that the absorption can be read out of the solution vector
     */
    PetscReal* sol = nullptr;          //!< The solution value at any given location
    PetscReal* temperature = nullptr;  //!< The temperature at any given location
    PetscScalar *rhsValues;
    DMPlexPointLocalFieldRead(fvmSolver.GetSubDomain().GetDM(), iCell, eulerFieldInfo.id, rhsArray, &rhsValues) >> utilities::PetscUtilities::checkError;

    /**
     * Get the absorption out of the solution vector
     */
    absorptivityFunction.function(sol, *temperature, &kappa, absorptivityFunctionContext);

    /**
     * Add the computed intensity to the energy equation
     */
    rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += GetIntensity(*temperature, kappa);  //!< Loop through the cells and update the equation of state
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
#define COMMA ,
REGISTER_PASS_THROUGH(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::RadiationLoss, "uses math functions to add arbitrary sources to the fvm method",
                      std::map<std::string COMMA ablate::mathFunctions::MathFunction>);
