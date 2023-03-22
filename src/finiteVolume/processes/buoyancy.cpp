#include "buoyancy.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
ablate::finiteVolume::processes::Buoyancy::Buoyancy(std::vector<double> buoyancyVector) : buoyancyVector(buoyancyVector) {}

void ablate::finiteVolume::processes::Buoyancy::Setup(ablate::finiteVolume::FiniteVolumeSolver &fv) {
    // Before each step, update the avg density
    auto buoyancyPreStep = std::bind(&ablate::finiteVolume::processes::Buoyancy::UpdateAverageDensity, this, std::placeholders::_1, std::placeholders::_2);
    fv.RegisterPreStep(buoyancyPreStep);

    // add the source function
    fv.RegisterRHSFunction(ComputeBuoyancySource, this, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::EULER_FIELD}, {});
}
PetscErrorCode ablate::finiteVolume::processes::Buoyancy::UpdateAverageDensity(TS flowTs, ablate::solver::Solver &flow) {
    PetscFunctionBeginUser;
    PetscReal locDensitySum = 0.0;
    PetscInt locCellCount = 0;

    // get access to the underlying data for the flow
    PetscInt flowEulerId = flow.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD).id;

    // get the flowSolution from the ts
    Vec globFlowVec = flow.GetSubDomain().GetSolutionVector();
    const PetscScalar *flowArray;
    PetscCall(VecGetArrayRead(globFlowVec, &flowArray));

    // Get the valid cell range over this region
    ablate::domain::Range cellRange;
    flow.GetCellRange(cellRange);

    // March over each cell
    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cellRange.points ? cellRange.points[c] : c;

        // Get the current state variables for this cell
        const PetscScalar *euler;
        PetscCall(DMPlexPointGlobalFieldRead(flow.GetSubDomain().GetDM(), cell, flowEulerId, flowArray, &euler));

        if (euler) {
            locDensitySum += euler[CompressibleFlowFields::RHO];
            locCellCount++;
        }
    }

    // sum across all mpi ranks
    PetscReal densitySum = 0.0;
    PetscInt cellCount = 0;
    auto comm = flow.GetSubDomain().GetComm();
    PetscCallMPI(MPIU_Allreduce(&locDensitySum, &densitySum, 1, MPIU_REAL, MPIU_SUM, comm));
    PetscCallMPI(MPIU_Allreduce(&locCellCount, &cellCount, 1, MPIU_INT, MPIU_SUM, comm));

    // update reference density
    densityAvg = densitySum / cellCount;

    // cleanup
    flow.RestoreRange(cellRange);
    PetscCall(VecRestoreArrayRead(globFlowVec, &flowArray));

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::Buoyancy::ComputeBuoyancySource(PetscInt dim, PetscReal time, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u,
                                                                                const PetscInt *aOff, const PetscScalar *a, PetscScalar *f, void *ctx) {
    PetscFunctionBeginUser;
    const int EULER_FIELD = 0;
    auto buoyancyProcess = (ablate::finiteVolume::processes::Buoyancy *)ctx;

    // exact some values
    const PetscReal density = u[uOff[EULER_FIELD] + CompressibleFlowFields::RHO];

    // set the source terms
    f[CompressibleFlowFields::RHO] = 0.0;
    f[CompressibleFlowFields::RHOE] = 0.0;

    // Add in the buoyancy source terms for momentum and energy
    for (PetscInt n = 0; n < dim; n++) {
        f[CompressibleFlowFields::RHOU + n] = PetscMax((density - buoyancyProcess->densityAvg) * buoyancyProcess->buoyancyVector[n], 0.0);
        PetscReal vel = u[uOff[EULER_FIELD] + CompressibleFlowFields::RHOU + n] / density;
        f[CompressibleFlowFields::RHOE] += vel * f[CompressibleFlowFields::RHOU + n];
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::Buoyancy, "build advection/diffusion for the euler field",
         ARG(std::vector<double>, "vector", "gravitational acceleration vector"));
