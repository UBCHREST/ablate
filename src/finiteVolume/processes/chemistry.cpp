#include "chemistry.hpp"

#include <utility>
#include "utilities/petscUtilities.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::processes::Chemistry::Chemistry(std::shared_ptr<ablate::eos::ChemistryModel> chemistryModel) : chemistryModel(std::move(chemistryModel)) {}

void ablate::finiteVolume::processes::Chemistry::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    // Before each step, compute the source term over the entire dt
    auto chemistryPreStage = std::bind(&ablate::finiteVolume::processes::Chemistry::ChemistryPreStage, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
    flow.RegisterPreStage(chemistryPreStage);

    // Add the rhs point function for the source
    flow.RegisterRHSFunction(AddChemistrySourceToFlow, this);
}

void ablate::finiteVolume::processes::Chemistry::Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    // determine the number of nodes we need to compute based upon the local solver
    ablate::domain::Range cellRange;
    flow.GetCellRangeWithoutGhost(cellRange);

    // size up a calculator for this number of fields and cell range
    sourceCalculator = chemistryModel->CreateSourceCalculator(flow.GetSubDomain().GetFields(), cellRange);

    flow.RestoreRange(cellRange);
}

PetscErrorCode ablate::finiteVolume::processes::Chemistry::ChemistryPreStage(TS flowTs, ablate::solver::Solver& solver, PetscReal stagetime) {
    PetscFunctionBegin;
    // get time step information from the ts
    PetscReal time;
    PetscCall(TSGetTime(flowTs, &time));

    // only continue if the stage time is the real time (i.e. the first stage)
    if (time != stagetime) {
        PetscFunctionReturn(0);
    }

    StartEvent("ChemistryPreStage");

    // Get the valid cell range over this region
    auto& fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver&>(solver);
    ablate::domain::Range cellRange;
    fvSolver.GetCellRangeWithoutGhost(cellRange);

    // store the current dt
    PetscReal dt;
    PetscCall(TSGetTimeStep(flowTs, &dt));

    // get the flowSolution from the ts
    Vec globFlowVec;
    PetscCall(TSGetSolution(flowTs, &globFlowVec));

    // Compute the current source terms
    try {
        sourceCalculator->ComputeSource(cellRange, time, dt, globFlowVec);
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exception.what());
    }

    // clean up
    solver.RestoreRange(cellRange);
    EndEvent();
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::Chemistry::AddChemistrySourceToFlow(const FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void* ctx) {
    PetscFunctionBegin;
    auto process = (ablate::finiteVolume::processes::Chemistry*)ctx;
    process->StartEvent("AddChemistrySourceToFlow");
    // get the cell range
    ablate::domain::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    // add in contributions
    try {
        process->sourceCalculator->AddSource(cellRange, locX, locFVec);
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exception.what());
    }

    // cleanup
    solver.RestoreRange(cellRange);

    process->EndEvent();
    PetscFunctionReturn(0);
}

void ablate::finiteVolume::processes::Chemistry::AddChemistrySourceToFlow(const FiniteVolumeSolver& solver, Vec locFVec) {
    AddChemistrySourceToFlow(solver, solver.GetSubDomain().GetDM(), NAN, nullptr, locFVec, this) >> utilities::PetscUtilities::checkError;
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::Chemistry, "adds chemistry source terms from a chemistry model to the finite volume flow",
         ARG(ablate::eos::ChemistryModel, "eos", "the eos/chemistry model to generate source terms"));
