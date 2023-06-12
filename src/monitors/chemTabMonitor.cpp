#include "chemTabMonitor.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"

#include <utility>
ablate::monitors::ChemTabMonitor::ChemTabMonitor(const std::shared_ptr<ablate::eos::EOS>& eos) : chemTabModel(std::dynamic_pointer_cast<ablate::eos::ChemTab>(eos)) {
    if (!chemTabModel) {
        throw std::invalid_argument("The ablate::monitors::ChemTabMonitor::ChemTabMonitor requires a ablate::eos::ChemTab model");
    }
}

void ablate::monitors::ChemTabMonitor::Register(std::shared_ptr<solver::Solver> solverIn) {
    Monitor::Register(solverIn);

    auto fvSolver = std::dynamic_pointer_cast<ablate::finiteVolume::FiniteVolumeSolver>(GetSolver());
    if (!fvSolver) {
        throw std::invalid_argument("The ablate::monitors::ChemTabMonitor::ChemTabMonitor requires a ablate::finiteVolume::FiniteVolumeSolver");
    }

    // register the update call
    fvSolver->RegisterPreRHSFunction(DecodeMassFractions, this);

    // Get the density function
    densityFunction = chemTabModel->GetThermodynamicFunction(eos::ThermodynamicProperty::Density, solverIn->GetSubDomain().GetFields());
}

PetscErrorCode ablate::monitors::ChemTabMonitor::DecodeMassFractions(ablate::finiteVolume::FiniteVolumeSolver& fvSolver, TS ts, PetscReal time, bool initialStage, Vec locXVec, void* ctx) {
    PetscFunctionBeginUser;
    auto chemTabMonitor = (ablate::monitors::ChemTabMonitor*)ctx;

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    fvSolver.GetCellRange(cellRange);

    // get the local aux vector and auxDM
    auto locAuxVec = fvSolver.GetSubDomain().GetAuxVector();
    auto dmAux = fvSolver.GetSubDomain().GetAuxDM();
    auto dm = fvSolver.GetSubDomain().GetDM();

    // extract the low flow and aux fields
    const PetscScalar* locXArray;
    VecGetArrayRead(locXVec, &locXArray) >> utilities::PetscUtilities::checkError;
    PetscScalar* locAuxArray;
    VecGetArray(locAuxVec, &locAuxArray) >> utilities::PetscUtilities::checkError;

    // Get the required field locations
    const auto& densityProgressField = fvSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD);
    const auto& yiField = fvSolver.GetSubDomain().GetField(eos::EOS::YI);
    auto densityFunction = chemTabMonitor->densityFunction.function;
    auto densityContext = chemTabMonitor->densityFunction.context.get();

    // March over each cell volume
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // Get the cell location
        const PetscInt cell = cellRange.GetPoint(c);

        const PetscReal* solutionValues;
        DMPlexPointLocalRead(dm, cell, locXArray, &solutionValues) >> utilities::PetscUtilities::checkError;

        PetscReal* yiValues;
        DMPlexPointLocalFieldRef(dmAux, cell, yiField.id, locAuxArray, &yiValues) >> utilities::PetscUtilities::checkError;

        if (solutionValues && yiValues) {
            // compute density
            PetscReal density;
            densityFunction(solutionValues, &density, densityContext) >> utilities::PetscUtilities::checkError;

            // Compute the yiValues from densityProgress field and density
            chemTabMonitor->chemTabModel->ComputeMassFractions(solutionValues + densityProgressField.offset, yiValues, density);
        }
    }

    VecRestoreArrayRead(locXVec, &locXArray) >> utilities::PetscUtilities::checkError;
    VecRestoreArray(locAuxVec, &locAuxArray) >> utilities::PetscUtilities::checkError;

    fvSolver.RestoreRange(cellRange);
    PetscFunctionReturn(0);
}

ablate::monitors::ChemTabMonitor::Fields::Fields(const std::shared_ptr<ablate::eos::EOS>& eos, std::shared_ptr<domain::Region> region)
    : chemTabModel(std::dynamic_pointer_cast<ablate::eos::ChemTab>(eos)), region(std::move(region)) {
    if (!chemTabModel) {
        throw std::invalid_argument("The ablate::monitors::ChemTabMonitor::Fields requires a ablate::eos::ChemTab model");
    }
}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::monitors::ChemTabMonitor::Fields::GetFields() {
    return {std::make_shared<domain::FieldDescription>(eos::EOS::YI, eos::EOS::YI, chemTabModel->GetSpeciesNames(), domain::FieldLocation::AUX, domain::FieldType::FVM, region)};
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::monitors::ChemTabMonitor::Fields, "Sets up the monitor field(s) needed by the ChemTabMonitor",
         ARG(ablate::eos::EOS, "eos", "must be a ablate::eos::ChemTab model"), OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"));

REGISTER(ablate::monitors::Monitor, ablate::monitors::ChemTabMonitor, "Update the yi aux field using the ChemTab model", ARG(ablate::eos::EOS, "eos", "must be a ablate::eos::ChemTab model"));