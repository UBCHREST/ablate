#include "compressibleFlow.h"


static const char *compressibleFlowFieldNames[TOTAL_COMPRESSIBLE_FLOW_FIELDS + 1] = {"density", "momentum", "energy", "unknown"};
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
    PetscReal totalEnergy = conservedValues[RHOE + dim-1]/(*density);

    // Get the velocity in this direction
    (*normalVelocity) = 0.0;
    for(PetscInt d =0; d < dim; d++){
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

 void CompressibleFlowComputeFluxRho(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(area[0]) > 1E-5) {
        // Compute the norm
        PetscReal norm[3];
        NormVector(dim, area, norm);

        // Decode the left and right states
        PetscReal densityL;
        PetscReal velocityNormalL;
        PetscReal velocityL[3];
        PetscReal internalEnergyL;
        PetscReal aL;
        PetscReal ML;
        PetscReal pL;
        DecodeState(dim, xL, norm, flowParameters->gamma, &densityL, &velocityNormalL,velocityL, &internalEnergyL, &aL, &ML, &pL);

        PetscReal densityR;
        PetscReal velocityNormalR;
        PetscReal velocityR[3];
        PetscReal internalEnergyR;
        PetscReal aR;
        PetscReal MR;
        PetscReal pR;
        DecodeState(dim, xR, norm, flowParameters->gamma, &densityR, &velocityNormalR, velocityR, &internalEnergyR, &aR, &MR, &pR);

        PetscReal sPm;
        PetscReal sPp;
        PetscReal sMm;
        PetscReal sMp;

        flowParameters->fluxDifferencer(MR, &sPm, &sMm, ML, &sPp, &sMp);

        // Compute M and P on the face
        PetscReal M = sMm + sMp;

        if(M < 0){
            // M- on Right
            flux[0] = M * densityR * aR * MagVector(dim, area);
        }else{
            // M+ on Left
            flux[0] = M * densityL * aL * MagVector(dim, area);
        }
    }else{
        flux[0] = 0.0;
    }
}

void CompressibleFlowComputeFluxRhoU(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(area[0]) > 1E-5) {
        // Compute the norm
        PetscReal norm[3];
        NormVector(dim, area, norm);
        const PetscReal areaMag = MagVector(dim, area);

        // Decode the left and right states
        PetscReal densityL;
        PetscReal velocityNormalL;
        PetscReal velocityL[3];
        PetscReal internalEnergyL;
        PetscReal aL;
        PetscReal ML;
        PetscReal pL;
        DecodeState(dim, xL, norm, flowParameters->gamma, &densityL, &velocityNormalL,velocityL, &internalEnergyL, &aL, &ML, &pL);

        PetscReal densityR;
        PetscReal velocityNormalR;
        PetscReal velocityR[3];
        PetscReal internalEnergyR;
        PetscReal aR;
        PetscReal MR;
        PetscReal pR;
        DecodeState(dim, xR, norm, flowParameters->gamma, &densityR, &velocityNormalR,velocityR,  &internalEnergyR, &aR, &MR, &pR);

        PetscReal sPm;
        PetscReal sPp;
        PetscReal sMm;
        PetscReal sMp;

        flowParameters->fluxDifferencer(MR, &sPm, &sMm, ML, &sPp, &sMp);

        // Compute M and P on the face
        PetscReal M = sMm + sMp;
        PetscReal p = pR*sPm + pL*sPp;

        // March over each component of momentum
        for(PetscInt n =0; n < dim; n++){
            if(M < 0){
                // M- on Right
                flux[n] = (M * densityR * aR * velocityR[n]) * areaMag + p*area[n];
            }else{
                // M+ on Left
                flux[n] = (M * densityL * aL * velocityL[n]) * areaMag + p*area[n];
            }
        }
    }else{
        flux[0] = 0.0;
        flux[1] = 0.0;
    }
}

void CompressibleFlowComputeFluxRhoE(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowData * flowParameters = (EulerFlowData *)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(area[0]) > 1E-5) {
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

        // Compute M and P on the face
        PetscReal M = sMm + sMp;

        if(M < 0){
            // M- on Right
            PetscReal velMag = MagVector(dim, velocityR);
            PetscReal HR = internalEnergyR + velMag*velMag/2.0 + pR/densityR;
            flux[0] = (M * densityR * aR * HR) * areaMag;
        }else{
            // M+ on Left
            PetscReal velMag = MagVector(dim, velocityL);
            PetscReal HL = internalEnergyL + velMag*velMag/2.0 + pL/densityL;
            flux[0] = (M * densityL * aL * HL) * areaMag;
        }
    }else{
        flux[0] = 0.0;
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

    // Register each field, this order must match the order in IncompressibleFlowFields enum
    ierr = FlowRegisterField(flowData, compressibleFlowFieldNames[RHO], compressibleFlowFieldNames[RHO], 1, FV);CHKERRQ(ierr);
    ierr = FlowRegisterField(flowData, compressibleFlowFieldNames[RHOU], compressibleFlowFieldNames[RHOU], dim, FV);CHKERRQ(ierr);
    ierr = FlowRegisterField(flowData, compressibleFlowFieldNames[RHOE], compressibleFlowFieldNames[RHOE], 1, FV);CHKERRQ(ierr);

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
    ierr = PetscDSSetRiemannSolver(prob, RHO, CompressibleFlowComputeFluxRho);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, RHO, flowData);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, RHOU,CompressibleFlowComputeFluxRhoU);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, RHOU, flowData);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, RHOE,CompressibleFlowComputeFluxRhoE);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, RHOE, flowData);CHKERRQ(ierr);
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
    Vec                cellgeom;
    ierr = DMPlexGetGeometryFVM(dm, NULL, &cellgeom, NULL);CHKERRQ(ierr);
    PetscInt cStart, cEnd;
    ierr = DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    DM dmCell;
    ierr = VecGetDM(cellgeom, &dmCell);CHKERRQ(ierr);
    const PetscScalar *cgeom;
    ierr = VecGetArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
    const PetscScalar      *x;
    ierr = VecGetArrayRead(v, &x);CHKERRQ(ierr);

    // March over volume
    PetscReal dtMin = 1.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        PetscFVCellGeom       *cg;
        const PetscReal           *xc;

        ierr = DMPlexPointLocalRead(dmCell, c, cgeom, &cg);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRead(dm, c, 0, x, &xc);CHKERRQ(ierr);

        if(xc) {  // must be real cell and not ghost

            PetscReal rho = xc[RHO];
            PetscReal u = xc[RHOU] / rho;
            PetscReal e = (xc[RHOE + 1] / rho) - 0.5 * u * u;//TODO: remove hard code
            PetscReal p = (flowParameters->gamma - 1) * rho * e;

            PetscReal a = PetscSqrtReal(flowParameters->gamma * p / rho);
            PetscReal dt = flowParameters->cfl * cg->volume / (a + PetscAbsReal(u));
            dtMin = PetscMin(dtMin, dt);
        }
    }
    PetscInt rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)ts), &rank);
    printf("dtMin(%d): %f\n", rank,dtMin );

    PetscReal dtMinGlobal;
    ierr = MPI_Allreduce(&dtMin, &dtMinGlobal, 1,MPIU_REAL, MPI_MIN, PetscObjectComm((PetscObject)ts));

    PetscPrintf(PetscObjectComm((PetscObject)ts), "TimeStep: %f\n", dtMinGlobal);
    ierr = TSSetTimeStep(ts, dtMinGlobal);CHKERRQ(ierr);

    if(PetscIsNanReal(dtMinGlobal)){
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_FP, "Invalid timestep selected for flow");
    }

    ierr = VecRestoreArrayRead(cellgeom, &cgeom);CHKERRQ(ierr);
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

//PetscErrorCode IncompressibleFlow_PackParameters(IncompressibleFlowParameters *parameters, PetscScalar *constantArray) {
//    constantArray[STROUHAL] = parameters->strouhal;
//    constantArray[REYNOLDS] = parameters->reynolds;
//    constantArray[PECLET] = parameters->peclet;
//    constantArray[MU] = parameters->mu;
//    constantArray[K] = parameters->k;
//    constantArray[CP] = parameters->cp;
//    return 0;
//}
//
//PetscErrorCode IncompressibleFlow_ParametersFromPETScOptions(PetscBag *flowParametersBag) {
//    IncompressibleFlowParameters *p;
//    PetscErrorCode ierr;
//
//    PetscFunctionBeginUser;
//    // create an empty bag
//    ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(IncompressibleFlowParameters), flowParametersBag);CHKERRQ(ierr);
//
//    // setup PETSc parameter bag
//    ierr = PetscBagGetData(*flowParametersBag, (void **)&p);CHKERRQ(ierr);
//    ierr = PetscBagSetName(*flowParametersBag, "flowParameters", "Low Mach Flow Parameters");CHKERRQ(ierr);
//    ierr = PetscBagRegisterReal(*flowParametersBag, &p->strouhal, 1.0, incompressibleFlowParametersTypeNames[STROUHAL], "Strouhal number");CHKERRQ(ierr);
//    ierr = PetscBagRegisterReal(*flowParametersBag, &p->reynolds, 1.0, incompressibleFlowParametersTypeNames[REYNOLDS], "Reynolds number");CHKERRQ(ierr);
//    ierr = PetscBagRegisterReal(*flowParametersBag, &p->peclet, 1.0, incompressibleFlowParametersTypeNames[PECLET], "Peclet number");CHKERRQ(ierr);
//    ierr = PetscBagRegisterReal(*flowParametersBag, &p->mu, 1.0, incompressibleFlowParametersTypeNames[MU], "non-dimensional viscosity");CHKERRQ(ierr);
//    ierr = PetscBagRegisterReal(*flowParametersBag, &p->k, 1.0, incompressibleFlowParametersTypeNames[K], "non-dimensional thermal conductivity");CHKERRQ(ierr);
//    ierr = PetscBagRegisterReal(*flowParametersBag, &p->cp, 1.0, incompressibleFlowParametersTypeNames[CP], "non-dimensional specific heat capacity");CHKERRQ(ierr);
//
//    PetscFunctionReturn(0);
//}