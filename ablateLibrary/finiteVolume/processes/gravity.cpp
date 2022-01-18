#include "gravity.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
ablate::finiteVolume::processes::Gravity::Gravity(std::vector<double> gravityVector) : gravityVector(gravityVector) {}

void ablate::finiteVolume::processes::Gravity::Initialize(ablate::finiteVolume::FiniteVolumeSolver &fv) {
    // Before each step, update the avg density
    auto gravityPreStep = std::bind(&ablate::finiteVolume::processes::Gravity::UpdateAverageDensity, this, std::placeholders::_1, std::placeholders::_2);
    fv.RegisterPreStep(gravityPreStep);

    // add the source function
    fv.RegisterRHSFunction(ComputeGravitySource, this, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::EULER_FIELD}, {});
}
PetscErrorCode ablate::finiteVolume::processes::Gravity::UpdateAverageDensity(TS flowTs, ablate::solver::Solver &flow) {
    PetscFunctionBeginUser;
    PetscReal locDensitySum = 0.0;
    PetscInt locCellCount = 0;

    // get access to the underlying data for the flow
    PetscInt flowEulerId = flow.GetSubDomain().GetField(finiteVolume::CompressibleFlowFields::EULER_FIELD).id;

    // get the flowSolution from the ts
    Vec globFlowVec = flow.GetSubDomain().GetSolutionVector();
    const PetscScalar *flowArray;
    PetscErrorCode ierr = VecGetArrayRead(globFlowVec, &flowArray);
    CHKERRQ(ierr);

    // Get the valid cell range over this region
    IS cellIS;
    PetscInt cStart, cEnd;
    const PetscInt *cells;
    flow.GetCellRange(cellIS, cStart, cEnd, cells);

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // Get the current state variables for this cell
        const PetscScalar *euler;
        ierr = DMPlexPointGlobalFieldRead(flow.GetSubDomain().GetDM(), cell, flowEulerId, flowArray, &euler);
        CHKERRQ(ierr);

        if (euler) {
            locDensitySum += euler[RHO];
            locCellCount++;
        }
    }

    // sum across all mpi ranks
    PetscReal densitySum = 0.0;
    PetscInt cellCount = 0;
    auto comm = flow.GetSubDomain().GetComm();
    ierr = MPIU_Allreduce(&locDensitySum, &densitySum, 1, MPIU_REAL, MPIU_SUM, comm);
    CHKERRMPI(ierr);
    ierr = MPIU_Allreduce(&locCellCount, &cellCount, 1, MPIU_INT, MPIU_SUM, comm);
    CHKERRMPI(ierr);

    // update reference density
    densityAvg = densitySum / cellCount;

    // cleanup
    flow.RestoreRange(cellIS, cStart, cEnd, cells);
    ierr = VecRestoreArrayRead(globFlowVec, &flowArray);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::Gravity::ComputeGravitySource(PetscInt dim, const PetscFVCellGeom *cg, const PetscInt *uOff, const PetscScalar *u, const PetscScalar *const *gradU,
                                                                              const PetscInt *aOff, const PetscScalar *a, const PetscScalar *const *gradA, PetscScalar *f, void *ctx) {
    PetscFunctionBeginUser;
    const int EULER_FIELD = 0;
    auto gravityProcess = (ablate::finiteVolume::processes::Gravity *)ctx;

    // exact some values
    const PetscReal density = u[uOff[EULER_FIELD] + RHO];

    // set the source terms
    f[RHO] = 0.0;
    f[RHOE] = 0.0;

    // Add in the Add buoyancy source terms for energy
    for (PetscInt n = 0; n < dim; n++) {
        f[RHOU] = PetscMax((density - gravityProcess->densityAvg) * gravityProcess->gravityVector[n], 0.0);
        PetscReal vel = u[uOff[EULER_FIELD] + RHOU + n] / density;
        f[RHOE] += vel * f[RHOU];
    }

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::Gravity, "build advection/diffusion for the euler field",
         ARG(std::vector<double>, "vector", "gravitational acceleration vector"));
