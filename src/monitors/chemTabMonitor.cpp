#include "chemTabMonitor.hpp"

#include "finiteVolume/compressibleFlowFields.hpp"

ablate::monitors::ChemTabMonitor::ChemTabMonitor(const std::shared_ptr<ablate::eos::ChemistryModel>& chemTabIn) : chemTab(std::dynamic_pointer_cast<eos::ChemTab>(chemTabIn)) {
    if (!chemTabIn) {
        throw std::invalid_argument("The ablate::monitors::ChemTabMonitor monitor can only be used with eos::ChemTab");
    }
}

void ablate::monitors::ChemTabMonitor::Register(std::shared_ptr<solver::Solver> solverIn) {
    // Name this monitor
    auto monitorName = solverIn->GetSolverId() + "_chemTab";

    // Define the required fields
    std::vector<std::shared_ptr<domain::FieldDescriptor>> fields{
        std::make_shared<domain::FieldDescription>(ablate::finiteVolume::CompressibleFlowFields::YI_FIELD,
                                                   ablate::finiteVolume::CompressibleFlowFields::YI_FIELD,
                                                   chemTab->GetSpeciesNames(),
                                                   domain::FieldLocation::SOL,
                                                   domain::FieldType::FVM),
        std::make_shared<domain::FieldDescription>("energySource", "energySource", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM),
        std::make_shared<domain::FieldDescription>("progressSource", "progressSource", chemTab->GetProgressVariables(), domain::FieldLocation::SOL, domain::FieldType::FVM)};

    // get the required function to compute density
    densityFunction = chemTab->GetThermodynamicFunction(eos::ThermodynamicProperty::Density, solverIn->GetSubDomain().GetFields());

    // this probe will only work with fV flow with a single mpi rank for now.  It should be replaced with DMInterpolationEvaluate
    auto finiteVolumeSolver = std::dynamic_pointer_cast<ablate::finiteVolume::FiniteVolumeSolver>(solverIn);
    if (!finiteVolumeSolver) {
        throw std::invalid_argument("The ablate::monitors::ChemTabMonitor monitor can only be used with ablate::finiteVolume::FiniteVolumeSolver");
    }

    // get a reference to the tchem reactions instance in the solver
    chemistry = finiteVolumeSolver->FindProcess<ablate::finiteVolume::processes::Chemistry>();

    // call the base function to create the domain
    FieldMonitor::Register(monitorName, solverIn, fields);
}

PetscErrorCode ablate::monitors::ChemTabMonitor::Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    PetscFunctionBeginUser;
    // get the required fields from the fieldDm and main dm
    const auto& yiMonitorField = monitorSubDomain->GetField(ablate::finiteVolume::CompressibleFlowFields::YI_FIELD);
    const auto& energySourceField = monitorSubDomain->GetField("energySource");
    const auto& progressSourceField = monitorSubDomain->GetField("progressSource");
    const auto& eulerField = GetSolver()->GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::EULER_FIELD);
    const auto& densityProgressField = GetSolver()->GetSubDomain().GetField(ablate::finiteVolume::CompressibleFlowFields::DENSITY_PROGRESS_FIELD);
    const auto& yiField = GetSolver()->GetSubDomain().GetField(ablate::eos::ChemTab::DENSITY_YI_DECODE_FIELD);

    // define a localFVec from the solution dm to compute the source terms
    Vec sourceTermVec = nullptr;
    if (chemistry) {
        PetscCall(DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &sourceTermVec));
        PetscCall(VecZeroEntries(sourceTermVec));
        auto fvSolver = std::dynamic_pointer_cast<ablate::finiteVolume::FiniteVolumeSolver>(GetSolver());

        // Get the local solution array
        Vec localSolutionVector;
        PetscCall(DMGetLocalVector(GetSolver()->GetSubDomain().GetDM(), &localSolutionVector));
        PetscCall(DMGlobalToLocal(GetSolver()->GetSubDomain().GetDM(), GetSolver()->GetSubDomain().GetSolutionVector(), INSERT_VALUES, localSolutionVector));
        chemistry->AddChemistrySourceToFlow(*fvSolver, localSolutionVector, sourceTermVec);
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
                monitorField[yiMonitorField.offset + sp] = solutionField[yiField.offset + sp] / density;
            }

            if (sourceTermField) {
                monitorField[energySourceField.offset] = sourceTermField[eulerField.offset + ablate::finiteVolume::CompressibleFlowFields::RHOE] / density;
                for (PetscInt s = 0; s < densityProgressField.numberComponents; ++s) {
                    monitorField[progressSourceField.offset + s] = sourceTermField[densityProgressField.offset + s] / density;
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
REGISTER(ablate::monitors::Monitor, ablate::monitors::ChemTabMonitor, "This class reports the output values for chemTab",
         ARG(ablate::eos::ChemistryModel, "eos", "the chemTab model used for the calculation"));