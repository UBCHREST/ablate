#include "compressibleFlow.h"

static const char *compressibleFlowFieldNames[TOTAL_COMPRESSIBLE_FLOW_FIELDS + 1] = {"density", "energy", "momentum", "unknown"};
static const char *incompressibleSourceFieldNames[TOTAL_COMPRESSIBLE_FLOW_PARAMETERS + 1] = {"cfl", "gamma", "unknown"};

static inline void NormVector(PetscInt dim, const PetscReal* in, PetscReal* out){
    PetscReal mag = 0.0;
    for (PetscInt d=0; d< dim; d++) {
        mag += in[d]*in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d=0; d< dim; d++) {
        out[d] = in[d]/mag;
    }
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal* in){
    PetscReal mag = 0.0;
    for (PetscInt d=0; d< dim; d++) {
        mag += in[d]*in[d];
    }
    return PetscSqrtReal(mag);
}

/**
 * Function to get the density, velocity, and energy from the conserved variables
 * @return
 */
static void DecodeState(PetscInt dim, const PetscReal* conservedValues,  const PetscReal *normal, PetscReal gamma, PetscReal* density,
                                  PetscReal* normalVelocity, PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p){

    // decode
    *density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE]/(*density);

    // Get the velocity in this direction
    (*normalVelocity) = 0.0;
    for (PetscInt d =0; d < dim; d++){
        velocity[d] = conservedValues[RHOU + d]/(*density);
        (*normalVelocity) += velocity[d]*normal[d];
    }

    // get the speed
    PetscReal speed = MagVector(dim, velocity);

    // assumed eos
    (*internalEnergy) = (totalEnergy) - 0.5 * speed * speed;
    *p = (gamma - 1.0)*(*density)*(*internalEnergy);
    *a = PetscSqrtReal(gamma*(*p)/(*density));

    *M = (*normalVelocity)/(*a);
}

void CompressibleFlowComputeEulerFlux(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, area, norm);
    const PetscReal areaMag = MagVector(dim, area);

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;
    DecodeState(dim, xL, norm, flowParameters->gamma, &densityL, &normalVelocityL, velocityL, &internalEnergyL, &aL, &ML, &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    DecodeState(dim, xR, norm, flowParameters->gamma, &densityR, &normalVelocityR, velocityR, &internalEnergyR, &aR, &MR, &pR);

    PetscReal sPm;
    PetscReal sPp;
    PetscReal sMm;
    PetscReal sMp;

    flowParameters->fluxDifferencer(MR, &sPm, &sMm, ML, &sPp, &sMp);

    flux[RHO] = (sMm* densityR * aR + sMp* densityL * aL) * areaMag;

    PetscReal velMagR = MagVector(dim, velocityR);
    PetscReal HR = internalEnergyR + velMagR*velMagR/2.0 + pR/densityR;
    PetscReal velMagL = MagVector(dim, velocityL);
    PetscReal HL = internalEnergyL + velMagL*velMagL/2.0 + pL/densityL;

    flux[RHOE] = (sMm * densityR * aR * HR + sMp * densityL * aL * HL) * areaMag;

    for (PetscInt n =0; n < dim; n++) {
        flux[RHOU + n] = (sMm * densityR * aR * velocityR[n] + sMp * densityL * aL * velocityL[n]) * areaMag + (pR*sPm + pL*sPp) * area[n];
    }
}

PetscErrorCode CompressibleFlow_SetupDiscretization(FlowData flowData, DM dm) {
    PetscInt dim;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    const PetscInt ghostCellDepth = 1;
    {// Make sure that the flow is setup distributed
        DM dmDist;
        // ierr = DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE);CHKERRQ(ierr);
        ierr = DMPlexDistribute(dm, ghostCellDepth, NULL, &dmDist);CHKERRQ(ierr);
        if (dmDist) {
            ierr = DMDestroy(&dm);CHKERRQ(ierr);
            dm   = dmDist;
        }
    }

    // create any ghost cells that are needed
    {
        DM gdm;
        ierr = DMPlexConstructGhostCells(dm, NULL, NULL, &gdm);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = gdm;
    }

    //Store the field data
    flowData->dm = dm;
    ierr = DMSetApplicationContext(flowData->dm, flowData);CHKERRQ(ierr);

    // Determine the number of dimensions
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // Register a single field
    ierr = FlowRegisterField(flowData, "euler", "euler", 2+dim, FV);CHKERRQ(ierr);

    // Create the discrete systems for the DM based upon the fields added to the DM
    ierr = FlowFinalizeRegisterFields(flowData);CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlow_StartProblemSetup(FlowData flowData, PetscInt num, PetscScalar values[]) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);CHKERRQ(ierr);

    // Set the flux calculator solver for each component
    ierr = PetscDSSetRiemannSolver(prob, 0, CompressibleFlowComputeEulerFlux);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, 0, flowData);CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);

    // Create the euler data
    EulerFlowData *data;
    PetscNew(&data);
    flowData->data =data;

    data->cfl = values[CFL];
    data->gamma = values[GAMMA];

    const char *prefix;
    ierr = DMGetOptionsPrefix(flowData->dm, &prefix);CHKERRQ(ierr);

    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)flowData->dm),prefix,"Compressible Flow Options",NULL);CHKERRQ(ierr);
    PetscFunctionList fluxDifferencerList;
    ierr = FluxDifferencerListGet(&fluxDifferencerList);CHKERRQ(ierr);
    char fluxDiffValue[128] = "ausm";
    ierr = PetscOptionsFList("-flux_diff","Flux differencer","",fluxDifferencerList,fluxDiffValue,fluxDiffValue,sizeof fluxDiffValue,NULL);CHKERRQ(ierr);
    ierr = FluxDifferencerGet(fluxDiffValue, &(data->fluxDifferencer));CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

static PetscErrorCode ComputeTimeStep(TS ts, void* context){
    PetscFunctionBeginUser;
    // Get the dm and current solution vector
    DM dm;
    PetscErrorCode ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    Vec v;
    TSGetSolution(ts, &v);

    // Get the flow param
    FlowData flowData;
    ierr = DMGetApplicationContext(dm, &flowData);CHKERRQ(ierr);
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;

    // Get the fv geom
    PetscReal minCellRadius;
    ierr = DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius);CHKERRQ(ierr);
    PetscInt cStart, cEnd;
    ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    const PetscScalar      *x;
    ierr = VecGetArrayRead(v, &x);CHKERRQ(ierr);

    //Get the dim from the dm
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 *minCellRadius;

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscReal           *xc;
        ierr = DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc);CHKERRQ(ierr);

        if (xc) {  // must be real cell and not ghost
            PetscReal rho = xc[RHO];

            // Compute the kinetic energy
            PetscReal velMag = 0.0;
            for (PetscInt i =0; i < dim; i++){
                velMag += PetscSqr(xc[RHOU + i] / rho);
            }

            PetscReal u = xc[RHOU] / rho;
            PetscReal e = (xc[RHOE] / rho) - 0.5 * velMag;
            PetscReal p = (flowParameters->gamma - 1) * rho * e;


            PetscReal a = PetscSqrtReal(flowParameters->gamma * p / rho);
            PetscReal dt = flowParameters->cfl * dx / (a + PetscAbsReal(u));
            dtMin = PetscMin(dtMin, dt);
        }
    }
    PetscInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);

    PetscReal dtMinGlobal;
    ierr = MPI_Allreduce(&dtMin, &dtMinGlobal, 1,MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts));

    ierr = TSSetTimeStep(ts, dtMinGlobal);CHKERRQ(ierr);

    if (PetscIsNanReal(dtMinGlobal)){
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_FP, "Invalid timestep selected for flow");
    }

    ierr = VecRestoreArrayRead(v, &x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode CompressibleFlow_CompleteProblemSetup(FlowData flowData, TS ts) {
    PetscErrorCode ierr;
    DM dm;

    PetscFunctionBeginUser;
    ierr = FlowCompleteProblemSetup(flowData, ts);CHKERRQ(ierr);
    ierr = FlowRegisterPreStep(flowData, ComputeTimeStep, flowData);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}