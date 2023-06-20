#include "mixtureFractionMonitor.hpp"

#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"

ablate::monitors::MixtureFractionMonitor::MixtureFractionMonitor(std::shared_ptr<MixtureFractionCalculator> mixtureFractionCalculator)
    : mixtureFractionCalculator(std::move(std::move(mixtureFractionCalculator))) {}

void ablate::monitors::MixtureFractionMonitor::Register(std::shared_ptr<solver::Solver> solverIn) {
    // Name this monitor
    auto monitorName = solverIn->GetSolverId() + "_mixtureFraction";

    // Define the required fields
    std::vector<std::shared_ptr<domain::FieldDescriptor>> fields{
        std::make_shared<domain::FieldDescription>("zMix", "zMix", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM),
        std::make_shared<domain::FieldDescription>(ablate::finiteVolume::CompressibleFlowFields::YI_FIELD,
                                                   ablate::finiteVolume::CompressibleFlowFields::YI_FIELD,
                                                   mixtureFractionCalculator->GetEos()->GetSpeciesVariables(),
                                                   domain::FieldLocation::SOL,
                                                   domain::FieldType::FVM),
        std::make_shared<domain::FieldDescription>("energySource", "energySource", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM),
        std::make_shared<domain::FieldDescription>("yiSource", "yiSource", mixtureFractionCalculator->GetEos()->GetSpeciesVariables(), domain::FieldLocation::SOL, domain::FieldType::FVM)};

    // get the required function to compute density
    densityFunction = mixtureFractionCalculator->GetEos()->GetThermodynamicFunction(eos::ThermodynamicProperty::Density, solverIn->GetSubDomain().GetFields());

    // this probe will only work with fV flow with a single mpi rank for now.  It should be replaced with DMInterpolationEvaluate
    auto finiteVolumeSolver = std::dynamic_pointer_cast<ablate::finiteVolume::FiniteVolumeSolver>(solverIn);
    if (!finiteVolumeSolver) {
        throw std::invalid_argument("The MixtureFractionMonitor monitor can only be used with ablate::finiteVolume::FiniteVolumeSolver");
    }
    // get a reference to the tchem reactions instance in the solver
    chemistry = finiteVolumeSolver->FindProcess<ablate::finiteVolume::processes::Chemistry>();

    // call the base function to create the domain
    FieldMonitor::Register(monitorName, solverIn, fields);
}
PetscErrorCode ablate::monitors::MixtureFractionMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    // get the required fields from the fieldDm and main dm
    const auto& zMixMonitorField = monitorSubDomain->GetField("zMix");
    const auto& yiMonitorField = monitorSubDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::YI_FIELD);
    const auto& energySourceField = monitorSubDomain->GetField("energySource");
    const auto& densityYiSourceField = monitorSubDomain->GetField("yiSource");
    const auto& eulerField = GetSolver()->GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto& densityYiField = GetSolver()->GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::DENSITY_YI_FIELD);

    // define a localFVec from the solution dm to compute the source terms
    Vec sourceTermVec = nullptr;
    if (chemistry) {
        PetscCall(DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &sourceTermVec));
        PetscCall(VecZeroEntries(sourceTermVec));
        auto fvSolver = std::dynamic_pointer_cast<ablate::finiteVolume::FiniteVolumeSolver>(GetSolver());
        chemistry->AddChemistrySourceToFlow(*fvSolver, sourceTermVec);
    }

    // Get the arrays for the global vectors
    const PetscScalar* solutionFieldArray;
    PetscScalar* monitorFieldArray;
    PetscCall(VecGetArrayRead(GetSolver()->GetSubDomain().GetSolutionVector(), &solutionFieldArray));
    PetscCall(VecGetArray(monitorSubDomain->GetSolutionVector(), &monitorFieldArray));

    // check for the tmpLocalFArray
    const PetscScalar* sourceTermArray = nullptr;
    if (sourceTermVec) {
        PetscCall(VecGetArrayRead(sourceTermVec, &sourceTermArray));
    }

    // March over each cell in the monitorDm
    PetscInt cStart, cEnd;
    PetscCall(DMPlexGetHeightStratum(monitorSubDomain->GetDM(), 0, &cStart, &cEnd));

    // Get the cells we need to march over
    DMLabel solutionToMonitor;
    PetscCall(DMPlexGetSubpointMap(monitorSubDomain->GetDM(), &solutionToMonitor));

    const PetscInt* monitorToSolution = nullptr;
    IS monitorToSolutionIs = nullptr;
    // if this is a submap, get the monitor to solution
    if (solutionToMonitor) {
        PetscCall(DMPlexGetSubpointIS(monitorSubDomain->GetDM(), &monitorToSolutionIs));
        PetscCall(ISGetIndices(monitorToSolutionIs, &monitorToSolution));
    }

    // save time to get densityFunctionContext
    const auto densityFunctionContext = densityFunction.context.get();

    for (PetscInt monitorPt = cStart; monitorPt < cEnd; ++monitorPt) {
        PetscInt solutionPt = monitorToSolution ? monitorToSolution[monitorPt] : monitorPt;

        // Get the solutionField and monitorField
        const PetscScalar* solutionField = nullptr;
        PetscCall(DMPlexPointGlobalRead(GetSolver()->GetSubDomain().GetDM(), solutionPt, solutionFieldArray, &solutionField));
        if (!solutionField) {
            continue;
        }

        PetscScalar* monitorField = nullptr;
        PetscCall(DMPlexPointGlobalRead(monitorSubDomain->GetDM(), monitorPt, monitorFieldArray, &monitorField));

        const PetscScalar* sourceTermField = nullptr;
        if (sourceTermArray) {
            PetscCall(DMPlexPointGlobalRead(GetSolver()->GetSubDomain().GetDM(), solutionPt, sourceTermArray, &sourceTermField));
        }
        // Do not bother in ghost cells
        if (monitorField) {
            // compute the density from the solutionPt
            PetscReal density;
            PetscCall(densityFunction.function(solutionField, &density, densityFunctionContext));

            // Copy over and compute yi
            for (PetscInt sp = 0; sp < yiMonitorField.numberComponents; sp++) {
                monitorField[yiMonitorField.offset + sp] = solutionField[densityYiField.offset + sp] / density;
            }

            // Compute mixture fraction
            monitorField[zMixMonitorField.offset] = mixtureFractionCalculator->Calculate(monitorField + yiMonitorField.offset);

            if (sourceTermField) {
                monitorField[energySourceField.offset] = sourceTermField[eulerField.offset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density;
                for (PetscInt s = 0; s < densityYiField.numberComponents; ++s) {
                    monitorField[densityYiSourceField.offset + s] = sourceTermField[densityYiField.offset + s] / density;
                }
            }
        }
    }

    // cleanup
    if (sourceTermVec) {
        PetscCall(VecRestoreArrayRead(sourceTermVec, &sourceTermArray));
        PetscCall(DMRestoreLocalVector(GetSolver()->GetSubDomain().GetDM(), &sourceTermVec));
    }
    if (monitorToSolutionIs) {
        PetscCall(ISRestoreIndices(monitorToSolutionIs, &monitorToSolution));
    }
    PetscCall(VecRestoreArrayRead(GetSolver()->GetSubDomain().GetSolutionVector(), &solutionFieldArray));
    PetscCall(VecRestoreArray(monitorSubDomain->GetSolutionVector(), &monitorFieldArray));

    // Call the base Save function only after the subdomain global function is updated
    PetscCall(FieldMonitor::Save(viewer, sequenceNumber, time));
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::MixtureFractionMonitor,
         "This class computes the mixture fraction for each point in the domain and outputs zMix, Yi, and source terms to the hdf5 file",
         ARG(ablate::monitors::MixtureFractionCalculator, "mixtureFractionCalculator", "the calculator used to compute zMix"));