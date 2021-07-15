#include "tChemReactions.hpp"
#include <utilities/petscError.hpp>

#if defined(PETSC_HAVE_TCHEM)
#if defined(MAX)
#undef MAX
#endif
#if defined(MIN)
#undef MIN
#endif
#include <TC_interface.h>
#include <TC_params.h>
#include <flow/processes/eulerAdvection.hpp>
#else
#error TChem is required for this example.  Reconfigure PETSc using --download-tchem.
#endif

ablate::flow::processes::TChemReactions::TChemReactions(std::shared_ptr<eos::TChem> eosIn)
    : fieldDm(nullptr),
      sourceVec(nullptr),
      eos(eosIn),
      numberSpecies(eosIn->GetSpecies().size()),
      ts(nullptr),
      pointData(nullptr),
      jacobian(nullptr),
      tchemScratch(nullptr),
      jacobianScratch(nullptr),
      rows(nullptr) {
    // size up the scratch variables
    PetscMalloc3(numberSpecies + 1, &tchemScratch, PetscSqr(numberSpecies + 1), &jacobianScratch, numberSpecies, &rows) >> checkError;
    // The rows will not change, so set them once
    for (std::size_t i = 0; i < numberSpecies + 1; i++) {
        rows[i] = i;
    }

    // Create a vector and mat for local ode calculation
    VecCreateSeq(PETSC_COMM_SELF, numberSpecies + 1, &pointData) >> checkError;
    MatCreateSeqDense(PETSC_COMM_SELF, numberSpecies + 1, numberSpecies + 1, NULL, &jacobian) >> checkError;
    MatSetFromOptions(jacobian) >> checkError;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Create timestepping solver context
              - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    TSCreate(PETSC_COMM_SELF, &ts) >> checkError;
    TSSetType(ts, TSARKIMEX) >> checkError;
    TSARKIMEXSetFullyImplicit(ts, PETSC_TRUE) >> checkError;
    TSARKIMEXSetType(ts, TSARKIMEX4) >> checkError;
    TSSetRHSFunction(ts, NULL, SinglePointChemistryRHS, this) >> checkError;
    TSSetRHSJacobian(ts, jacobian, jacobian, SinglePointChemistryJacobian, this) >> checkError;
    TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER) >> checkError;

    // set the adapting control
    TSSetSolution(ts, pointData) >> checkError;
    PetscReal dt = 1e-10; /* Initial time step */
    TSSetTimeStep(ts, dt) >> checkError;
    TSAdapt adapt;
    TSGetAdapt(ts, &adapt) >> checkError;
    TSAdaptSetStepLimits(adapt, 1e-12, 1E-4) >> checkError; /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
    TSSetMaxSNESFailures(ts, -1) >> checkError;             /* Retry step an unlimited number of times */
    TSSetFromOptions(ts) >> checkError;
}
ablate::flow::processes::TChemReactions::~TChemReactions() {
    if (fieldDm) {
        DMDestroy(&fieldDm) >> checkError;
    }
    if (sourceVec) {
        VecDestroy(&sourceVec) >> checkError;
    }
    if (ts) {
        TSDestroy(&ts) >> checkError;
    }
    if (pointData) {
        VecDestroy(&pointData) >> checkError;
    }
    if (jacobian) {
        MatDestroy(&jacobian) >> checkError;
    }
    PetscFree3(tchemScratch, jacobianScratch, rows) >> checkError;
}

void ablate::flow::processes::TChemReactions::Initialize(ablate::flow::FVFlow& flow) {
    // Create a copy of the dm for the solver
    DM coordDM;
    DMGetCoordinateDM(flow.GetDM(), &coordDM) >> checkError;
    DMClone(flow.GetDM(), &fieldDm) >> checkError;
    DMSetCoordinateDM(fieldDm, coordDM) >> checkError;
    PetscInt dim;
    DMGetDimension(fieldDm, &dim) >> checkError;

    // Setup the unknown field in the dm.  This is a single field that holds sources for rho, rho*E, rho*U, (rho*V, rho*W), Yi, Y1+1, Y1+n
    PetscFV fvm;
    PetscFVCreate(PetscObjectComm((PetscObject)fieldDm), &fvm) >> checkError;
    PetscObjectSetName((PetscObject)fvm, "chemistrySource") >> checkError;
    PetscFVSetFromOptions(fvm) >> checkError;
    PetscFVSetNumComponents(fvm, ablate::flow::processes::EulerAdvection::RHOU + dim + numberSpecies) >> checkError;
    DMAddField(fieldDm, NULL, (PetscObject)fvm) >> checkError;
    PetscFVDestroy(&fvm) >> checkError;

    // create a vector to hold the source terms
    DMCreateLocalVector(fieldDm, &sourceVec) >> checkError;

    // Before each step, compute the source term over the entire dt
    auto chemistryPreStep = std::bind(&ablate::flow::processes::TChemReactions::ChemistryFlowPreStep, this, std::placeholders::_1, std::placeholders::_2);
    flow.RegisterPreStep(chemistryPreStep);

    // Add the rhs point function for the source
    flow.RegisterRHSFunction(AddChemistrySourceToFlow, this);
}

PetscErrorCode ablate::flow::processes::TChemReactions::SinglePointChemistryRHS(TS ts, PetscReal t, Vec X, Vec F, void* ptr) {
    ablate::flow::processes::TChemReactions* solver = (ablate::flow::processes::TChemReactions*)ptr;
    PetscErrorCode ierr;
    PetscScalar* fArray;
    const PetscScalar* xArray;

    PetscFunctionBeginUser;
    ierr = VecGetArrayRead(X, &xArray);
    CHKERRQ(ierr);
    ierr = VecGetArray(F, &fArray);
    CHKERRQ(ierr);

    // copy over the XVec to the scratch variable for now
    ierr = PetscArraycpy(solver->tchemScratch, xArray, solver->numberSpecies + 1);
    CHKERRQ(ierr);

    // get the source (assuming constant pressure/mass)
    ierr = TC_getSrc(solver->tchemScratch, solver->numberSpecies + 1, fArray);
    TCCHKERRQ(ierr);

    ierr = VecRestoreArrayRead(X, &xArray);
    CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &fArray);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::flow::processes::TChemReactions::SinglePointChemistryJacobian(TS ts, PetscReal t, Vec X, Mat aMat, Mat pMat, void* ptr) {
    ablate::flow::processes::TChemReactions* solver = (ablate::flow::processes::TChemReactions*)ptr;
    PetscErrorCode ierr;
    const PetscInt nEeq = solver->numberSpecies + 1;

    PetscFunctionBeginUser;
    // copy over the XVec to the scratch variable for now
    const PetscScalar* xArray;
    ierr = VecGetArrayRead(X, &xArray);
    CHKERRQ(ierr);
    ierr = PetscArraycpy(solver->tchemScratch, xArray, nEeq);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(X, &xArray);
    CHKERRQ(ierr);

    // compute the analytical jacobian assuming constant pressure
    ierr = TC_getJacTYN(solver->tchemScratch, solver->numberSpecies, solver->jacobianScratch, 1);
    CHKERRQ(ierr);

    // Load the matrix
    ierr = MatSetOption(pMat, MAT_ROW_ORIENTED, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = MatSetOption(pMat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);
    CHKERRQ(ierr);
    ierr = MatZeroEntries(pMat);
    CHKERRQ(ierr);
    ierr = MatSetValues(pMat, nEeq, solver->rows, nEeq, solver->rows, solver->jacobianScratch, INSERT_VALUES);
    CHKERRQ(ierr);
    ierr = MatAssemblyBegin(pMat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(pMat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    if (aMat != pMat) {
        ierr = MatAssemblyBegin(aMat, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
        ierr = MatAssemblyEnd(aMat, MAT_FINAL_ASSEMBLY);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::processes::TChemReactions::ChemistryFlowPreStep(TS flowTs, ablate::flow::Flow& flow) {
    PetscInt stepNumber;
    TSGetStepNumber(flowTs, &stepNumber);
    PetscReal time;
    TSGetTime(flowTs, &time);
    PetscErrorCode ierr;

    PetscFunctionBegin;
    IS cellIS;
    DM plex;
    PetscInt depth;
    ierr = DMConvert(flow.GetDM(), DMPLEX, &plex);
    CHKERRQ(ierr);
    ierr = DMPlexGetDepth(plex, &depth);
    CHKERRQ(ierr);
    ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);
    CHKERRQ(ierr);
    if (!cellIS) {
        ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);
        CHKERRQ(ierr);
    }

    // Get the sell range
    PetscInt cStart, cEnd;
    const PetscInt* cells = NULL;
    ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);
    CHKERRQ(ierr);

    // get the dim
    PetscInt dim;
    ierr = DMGetDimension(flow.GetDM(), &dim);
    CHKERRQ(ierr);

    // store the current dt
    PetscReal dt;
    ierr = TSGetTimeStep(flowTs, &dt);
    CHKERRQ(ierr);

    // get access to the underlying data for the flow
    PetscInt flowEulerId = flow.GetFieldId("euler").value();
    PetscInt flowDensityYiId = flow.GetFieldId("densityYi").value();

    // get the flowSolution from the ts
    Vec globFlowVec;
    ierr = TSGetSolution(flowTs, &globFlowVec);
    CHKERRQ(ierr);
    const PetscScalar* flowArray;
    ierr = VecGetArrayRead(globFlowVec, &flowArray);
    CHKERRQ(ierr);

    // Get access to the chemistry source.  This is sized for euler + nspec
    PetscScalar* sourceArray;
    ierr = VecGetArray(sourceVec, &sourceArray);
    CHKERRQ(ierr);

    // store the eos temperature functions
    eos::ComputeTemperatureFunction temperatureFunction = eos->GetComputeTemperatureFunction();
    void* temperatureContext = eos->GetComputeTemperatureContext();

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // Get the current state variables for this cell
        const PetscScalar* euler;
        const PetscScalar* densityYi;
        ierr = DMPlexPointGlobalFieldRead(flow.GetDM(), cell, flowEulerId, flowArray, &euler);
        CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRead(flow.GetDM(), cell, flowDensityYiId, flowArray, &densityYi);
        CHKERRQ(ierr);

        // If a real cell (not ghost)
        if (euler) {
            // store the data for the chemistry ts (T, Yi...)
            PetscReal temperature;
            ierr = temperatureFunction(dim,
                                       euler[ablate::flow::processes::EulerAdvection::RHO],
                                       euler[ablate::flow::processes::EulerAdvection::RHOE] / euler[ablate::flow::processes::EulerAdvection::RHO],
                                       euler + ablate::flow::processes::EulerAdvection::RHOU,
                                       densityYi,
                                       &temperature,
                                       temperatureContext);
            CHKERRQ(ierr);

            // get access to the point ode solver
            PetscScalar* pointArray;
            ierr = VecGetArray(pointData, &pointArray);
            CHKERRQ(ierr);
            pointArray[0] = temperature;
            for (std::size_t s = 0; s < numberSpecies; s++) {
                pointArray[s + 1] = densityYi[s] / euler[ablate::flow::processes::EulerAdvection::RHO];
            }

            // precompute some values with the point array
            double mwMix;  // This is kinda of a hack, just pass in the tempYi working array while skipping the first index
            int err = TC_getMs2Wmix(pointArray + 1, numberSpecies, &mwMix);
            TCCHKERRQ(err);
            ierr = VecRestoreArray(pointData, &pointArray);
            CHKERRQ(ierr);

            // compute the pressure as this node from T, Yi
            double R = 1000.0 * RUNIV / mwMix;
            PetscReal pressure = euler[ablate::flow::processes::EulerAdvection::RHO] * temperature * R;
            TC_setThermoPres(pressure);

            // Do a soft reset on the ode solver
            ierr = TSSetTime(ts, time);
            CHKERRQ(ierr);
            ierr = TSSetMaxTime(ts, time + dt);
            CHKERRQ(ierr);

            // solver for this point
            ierr = TSSolve(ts, pointData);

            if(ierr != 0){
                std::cout << "Could not solve chemistry ode, setting source terms to zero" << std::endl;
                // Use the updated values to compute the source terms for euler and species transport
                PetscScalar* fieldSource;
                ierr = DMPlexPointLocalRef(fieldDm, cell, sourceArray, &fieldSource);
                CHKERRQ(ierr);

                fieldSource[ablate::flow::processes::EulerAdvection::RHO] = 0.0;
                fieldSource[ablate::flow::processes::EulerAdvection::RHOE] = 0.0;
                for (PetscInt d = 0; d < dim; d++) {
                    fieldSource[ablate::flow::processes::EulerAdvection::RHOU + d] = 0.0;
                }
                for (std::size_t sp = 0; sp < numberSpecies; sp++) {
                    // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
                    fieldSource[ablate::flow::processes::EulerAdvection::RHOU + dim + sp] = 0.0;
                }

                continue;
            }


            // Use the updated values to compute the source terms for euler and species transport
            PetscScalar* fieldSource;
            ierr = DMPlexPointLocalRef(fieldDm, cell, sourceArray, &fieldSource);
            CHKERRQ(ierr);

            // get the array data again
            VecGetArray(pointData, &pointArray) >> checkError;

            // Use the point array to compute the updatedInternalEnergy
            err = TC_getMs2Wmix(pointArray + 1, numberSpecies, &mwMix);
            TCCHKERRQ(err);
            PetscReal updatedInternalEnergy;
            ierr = eos::TChem::ComputeSensibleInternalEnergy(numberSpecies, pointArray, mwMix, updatedInternalEnergy);
            CHKERRQ(ierr);

            // compute the ke
            PetscReal ke = 0.0;
            for (PetscInt d = 0; d < dim; d++) {
                ke += PetscSqr(euler[ablate::flow::processes::EulerAdvection::RHOU + d] / euler[ablate::flow::processes::EulerAdvection::RHO]);
            }
            ke *= 0.5;

            // store the computed source terms
            fieldSource[ablate::flow::processes::EulerAdvection::RHO] = 0.0;
            fieldSource[ablate::flow::processes::EulerAdvection::RHOE] =
                (euler[ablate::flow::processes::EulerAdvection::RHO] * (updatedInternalEnergy + ke) - euler[ablate::flow::processes::EulerAdvection::RHOE]) / dt;
            for (PetscInt d = 0; d < dim; d++) {
                fieldSource[ablate::flow::processes::EulerAdvection::RHOU + d] = 0.0;
            }
            for (std::size_t sp = 0; sp < numberSpecies; sp++) {
                // for constant density problem, d Yi rho/dt = rho * d Yi/dt + Yi*d rho/dt = rho*dYi/dt ~~ rho*(Yi+1 - Y1)/dt
                fieldSource[ablate::flow::processes::EulerAdvection::RHOU + dim + sp] = (euler[ablate::flow::processes::EulerAdvection::RHO] * pointArray[sp + 1] - densityYi[sp]) / dt;
            }
            VecRestoreArray(pointData, &pointArray);
        }
    }

    // cleanup
    ierr = VecRestoreArray(sourceVec, &sourceArray);
    CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(globFlowVec, &flowArray);
    CHKERRQ(ierr);
    ierr = DMDestroy(&plex);
    CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::processes::TChemReactions::AddChemistrySourceToFlow(DM dm, PetscReal time, Vec locX, Vec fVec, void* ctx) {
    IS cellIS;
    DM plex;
    PetscInt depth;
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = DMConvert(dm, DMPLEX, &plex);
    CHKERRQ(ierr);
    ierr = DMPlexGetDepth(plex, &depth);
    CHKERRQ(ierr);
    ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);
    CHKERRQ(ierr);
    if (!cellIS) {
        ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);
        CHKERRQ(ierr);
    }

    // get the cell range
    PetscInt cStart, cEnd;
    const PetscInt* cells = NULL;
    ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);
    CHKERRQ(ierr);

    // get the dm for this
    PetscDS ds = NULL;
    ierr = DMGetCellDS(dm, cells ? cells[cStart] : cStart, &ds);
    CHKERRQ(ierr);

    // get access to the fArray
    PetscScalar* fArray;
    ierr = VecGetArray(fVec, &fArray);
    CHKERRQ(ierr);

    // hard code assuming only euler and density
    PetscInt totDim;
    ierr = PetscDSGetTotalDimension(ds, &totDim);
    CHKERRQ(ierr);

    // get access to the source array in the solver
    ablate::flow::processes::TChemReactions* solver = (ablate::flow::processes::TChemReactions*)ctx;
    const PetscScalar* sourceArray;
    ierr = VecGetArrayRead(solver->sourceVec, &sourceArray);
    CHKERRQ(ierr);

    // March over each cell
    for (PetscInt c = cStart; c < cEnd; ++c) {
        // if there is a cell array, use it, otherwise it is just c
        const PetscInt cell = cells ? cells[c] : c;

        // read the global f
        PetscScalar* rhs;
        ierr = DMPlexPointGlobalRef(dm, cell, fArray, &rhs);
        CHKERRQ(ierr);

        // if a real cell
        if (rhs) {
            // read the source from the local calc
            const PetscScalar* source;
            ierr = DMPlexPointLocalRead(solver->fieldDm, cell, sourceArray, &source);
            CHKERRQ(ierr);

            // copy over and add to rhs
            for (PetscInt d = 0; d < totDim; d++) {
                rhs[d] += source[d];
            }
            CHKERRQ(ierr);
        }
    }

    ierr = VecRestoreArray(fVec, &fArray);
    CHKERRQ(ierr);
    ierr = VecGetArrayRead(solver->sourceVec, &sourceArray);
    CHKERRQ(ierr);
    ierr = ISDestroy(&cellIS);
    CHKERRQ(ierr);
    ierr = DMDestroy(&plex);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::processes::FlowProcess, ablate::flow::processes::TChemReactions, "reactions using the TChem v1 library", OPT(eos::TChem, "eos", "the tChem v1 eos"));
