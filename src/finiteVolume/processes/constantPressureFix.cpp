#include "constantPressureFix.hpp"

#include <utility>
#include "eos/chemTab.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::finiteVolume::processes::ConstantPressureFix::ConstantPressureFix(std::shared_ptr<eos::EOS> eos, double pressure) : pressure(pressure), eos(std::move(eos)) {}

void ablate::finiteVolume::processes::ConstantPressureFix::Setup(ablate::finiteVolume::FiniteVolumeSolver& fv) {
    // use the eos to set up the eos functions
    densityFunction = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Density, fv.GetSubDomain().GetFields());
    internalSensibleEnergyFunction = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::InternalSensibleEnergy, fv.GetSubDomain().GetFields());

    eulerFromEnergyAndPressure =
        eos->GetFieldFunctionFunction(finiteVolume::CompressibleFlowFields::EULER_FIELD, eos::ThermodynamicProperty::InternalSensibleEnergy, eos::ThermodynamicProperty::Pressure);

    // this function is not needed if density_yi is not a field
    if (fv.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD)) {
        densityYiFromEnergyAndPressure =
            eos->GetFieldFunctionFunction(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD, eos::ThermodynamicProperty::InternalSensibleEnergy, eos::ThermodynamicProperty::Pressure);
    }

    if (std::dynamic_pointer_cast<eos::ChemTab>(eos)) {
        fv.RegisterPostEvaluate([this](TS ts, ablate::solver::Solver& fvSolver) {
            // get access to the underlying data for the flow
            const auto& eulerId = fvSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD);
            auto chemTab = std::dynamic_pointer_cast<eos::ChemTab>(eos);
            PetscInt densityProgressOffset = PETSC_DECIDE;
            PetscInt numberSpecies = chemTab->GetSpecies().size();
            PetscInt numberProgressVariables = chemTab->GetSpecies().size();

            if (fvSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD)) {
                const auto& densityProgressId = fvSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD);
                numberProgressVariables = densityProgressId.numberComponents;
                densityProgressOffset = densityProgressId.offset;
            }
            PetscInt dim = fvSolver.GetSubDomain().GetDimensions();

            // Size up an array for yi and velocity
            std::vector<PetscReal> yiScratch(numberSpecies);
            PetscReal velocityScratch[3];

            // Get the valid cell range over this region
            solver::Range cellRange;
            fvSolver.GetCellRange(cellRange);

            // get the solution vec
            auto solutionVec = fvSolver.GetSubDomain().GetSolutionVector();
            PetscScalar* solutionArray;
            VecGetArray(solutionVec, &solutionArray) >> checkError;

            auto dm = fvSolver.GetSubDomain().GetDM();

            // March over each cell
            for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
                // if there is a cell array, use it, otherwise it is just c
                const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

                PetscScalar* conservedValues = nullptr;
                DMPlexPointGlobalRead(dm, cell, solutionArray, &conservedValues);

                // Get the original density
                if (conservedValues) {
                    PetscReal originalDensity;
                    densityFunction.function(conservedValues, &originalDensity, densityFunction.context.get()) >> checkError;

                    // compute the current sensibleInternalEnergy
                    PetscReal internalSensibleEnergy;
                    internalSensibleEnergyFunction.function(conservedValues, &internalSensibleEnergy, internalSensibleEnergyFunction.context.get()) >> checkError;

                    // compute the current velocity and species (if provided)
                    for (PetscInt n = 0; n < dim; n++) {
                        velocityScratch[n] = conservedValues[eulerId.offset + finiteVolume::CompressibleFlowFields::RHOU + n] / originalDensity;
                    }

                    // compute the current mass fractions from chemTab
                    chemTab->ComputeMassFractions(conservedValues + densityProgressOffset, numberProgressVariables, yiScratch.data(), yiScratch.size(), originalDensity);

                    // use the eos to compute the new conserved forms based upon the current internal energy, velocity, yi and a constant pressure
                    eulerFromEnergyAndPressure(internalSensibleEnergy, pressure, dim, velocityScratch, yiScratch.data(), conservedValues + eulerId.offset);

                    // Get the updated density
                    PetscReal updatedDensity = conservedValues[eulerId.offset + finiteVolume::CompressibleFlowFields::RHO];

                    // scale the progress variables
                    for (PetscInt p = 0; p < numberProgressVariables; p++) {
                        conservedValues[densityProgressOffset + p] *= updatedDensity / originalDensity;
                    }
                }
            }

            // cleanup
            VecRestoreArray(solutionVec, &solutionArray) >> checkError;
            fvSolver.RestoreRange(cellRange);
        });

    } else {
        fv.RegisterPostEvaluate([this](TS ts, ablate::solver::Solver& fvSolver) {
            // get access to the underlying data for the flow
            const auto& eulerId = fvSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD);
            PetscInt densityYiOffset = PETSC_DECIDE;
            PetscInt numberSpecies = 0;
            if (fvSolver.GetSubDomain().ContainsField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD)) {
                const auto& densityYiId = fvSolver.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);
                numberSpecies = densityYiId.numberComponents;
                densityYiOffset = densityYiId.offset;
            }
            PetscInt dim = fvSolver.GetSubDomain().GetDimensions();

            // Size up an array for yi and velocity
            std::vector<PetscReal> yiScratch(numberSpecies);
            PetscReal velocityScratch[3];

            // Get the valid cell range over this region
            solver::Range cellRange;
            fvSolver.GetCellRange(cellRange);

            // get the solution vec
            auto solutionVec = fvSolver.GetSubDomain().GetSolutionVector();
            PetscScalar* solutionArray;
            VecGetArray(solutionVec, &solutionArray) >> checkError;

            auto dm = fvSolver.GetSubDomain().GetDM();

            // March over each cell
            for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
                // if there is a cell array, use it, otherwise it is just c
                const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

                PetscScalar* conservedValues = nullptr;
                DMPlexPointGlobalRead(dm, cell, solutionArray, &conservedValues);

                // Get the original density
                if (conservedValues) {
                    PetscReal originalDensity;
                    densityFunction.function(conservedValues, &originalDensity, densityFunction.context.get()) >> checkError;

                    // compute the current sensibleInternalEnergy
                    PetscReal internalSensibleEnergy;
                    internalSensibleEnergyFunction.function(conservedValues, &internalSensibleEnergy, internalSensibleEnergyFunction.context.get()) >> checkError;

                    // compute the current velocity and species (if provided)
                    for (PetscInt n = 0; n < dim; n++) {
                        velocityScratch[n] = conservedValues[eulerId.offset + finiteVolume::CompressibleFlowFields::RHOU + n] / originalDensity;
                    }
                    for (PetscInt s = 0; s < numberSpecies; s++) {
                        yiScratch[s] = conservedValues[densityYiOffset + s] / originalDensity;
                    }

                    // use the eos to compute the new conserved forms based upon the current internal energy, velocity, yi and a constant pressure
                    eulerFromEnergyAndPressure(internalSensibleEnergy, pressure, dim, velocityScratch, yiScratch.data(), conservedValues + eulerId.offset);
                    densityYiFromEnergyAndPressure(internalSensibleEnergy, pressure, dim, velocityScratch, yiScratch.data(), conservedValues + densityYiOffset);
                }
            }

            // cleanup
            VecRestoreArray(solutionVec, &solutionArray) >> checkError;
            fvSolver.RestoreRange(cellRange);
        });
    }
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::ConstantPressureFix, "Reset the density based upon a constant pressure value",
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"), ARG(double, "pressure", "the constant pressure value"));
