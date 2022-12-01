#include "chemistry.hpp"

#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/petscError.hpp"
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
    solver::Range cellRange;
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

    // Get the valid cell range over this region
    auto& fvSolver = dynamic_cast<ablate::finiteVolume::FiniteVolumeSolver&>(solver);
    solver::Range cellRange;
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
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::Chemistry::AddChemistrySourceToFlow(const FiniteVolumeSolver& solver, DM dm, PetscReal time, Vec locX, Vec locFVec, void* ctx) {
    PetscFunctionBegin;
    auto process = (ablate::finiteVolume::processes::Chemistry*)ctx;

    // get the cell range
    solver::Range cellRange;
    solver.GetCellRangeWithoutGhost(cellRange);

    // add in contributions
    try {
        process->sourceCalculator->AddSource(cellRange, locFVec);
    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exception.what());
    }

    // cleanup
    solver.RestoreRange(cellRange);

    PetscFunctionReturn(0);
}

void ablate::finiteVolume::processes::Chemistry::AddChemistrySourceToFlow(const FiniteVolumeSolver& solver, Vec locFVec) {
    AddChemistrySourceToFlow(solver, solver.GetSubDomain().GetDM(), NAN, nullptr, locFVec, this) >> checkError;
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::Chemistry, "adds chemistry source terms from a chemistry model to the finite volume flow",
         ARG(ablate::eos::ChemistryModel, "eos", "the eos/chemistry model to generate source terms"));
