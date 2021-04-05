#include "compressibleFlow.h"


static const char *compressibleFlowFieldNames[TOTAL_COMPRESSIBLE_FLOW_FIELDS + 1] = {"density", "momentum", "energy", "unknown"};

// Parameters alpha and beta
// - Give improved results over AUSM and results are comparable to Roe splitting
// - Reference: "A Sequel to AUSM: AUSM+", Liou, pg 368, Eqn (22a, 22b), 1996
const static PetscReal AUSMbeta  = 1.e+0 / 8.e+0;
const static PetscReal AUSMalpha = 3.e+0 / 16.e+0;

typedef struct{
    PetscReal f;
    PetscReal fprm;
} PressureFunction;

static PressureFunction f_and_fprm_rarefaction(PetscReal pstar, PetscReal pLR, PetscReal aLR, PetscReal gam, PetscReal gamm1,PetscReal gamp1) {
    // compute value of pressure function for rarefaction
    PressureFunction function;
    function.f = ((2. * aLR) / gamm1) * (pow(pstar / pLR, 0.5 * gamm1 / gam) - 1.);
    function.fprm = (aLR / pLR / gam) * pow(pstar / pLR, -0.5 * gamp1 / gam);
    return function;
}

static PressureFunction f_and_fprm_shock(PetscReal pstar, PetscReal pLR, PetscReal rhoLR, PetscReal gam, PetscReal gamm1, PetscReal gamp1){
    // compute value of pressure function for shock
    PetscReal A = 2./gamp1/rhoLR;
    PetscReal B = gamm1*pLR/gamp1;
    PetscReal sqrtterm =  PetscSqrtReal(A/(pstar+B));
    PressureFunction function;
    function.f = (pstar-pLR)*sqrtterm;
    function.fprm = sqrtterm*(1.-0.5*(pstar-pLR)/(B+pstar));
    return function;
}


/*
 * Returns the plus split Mach number (+) using Van Leer splitting
 * - Reference 1: "A New Flux Splitting Scheme" Liou and Steffen, pg 26, Eqn (6), 1993
 * - Reference 2: "A Sequel to AUSM: AUSM+" Liou, pg 366, Eqn (8), 1996, actually eq. 19a
 * - Reference 3: "A sequel to AUSM, Part II: AUSM+-up for all speeds" Liou, pg 141, Eqn (18), 2006
 * - Capital script M_(1) in this reference
 */
static PetscReal sM1p (PetscReal M) {
    // Equation: 1/2*[M+|M|]
    return (0.5*(M+PetscAbsReal(M)));
}

/*
 * Returns the minus split Mach number (-) using Van Leer splitting
 * - Reference 1: "A New Flux Splitting Scheme" Liou and Steffen, pg 26, Eqn (6), 1993
 * - Reference 2: "A Sequel to AUSM: AUSM+" Liou, pg 366, Eqn (8), 1996
 * - Reference 3: "A sequel to AUSM, Part II: AUSM+-up for all speeds" Liou, pg 141, Eqn (18), 2006
 * - Capital script M_(1) in this reference
 */
static PetscReal sM1m (PetscReal M) {
    // Equation: 1/2*[M-|M|]
    return (0.5*(M-PetscAbsReal(M)));
}
//
///*
// * Computes the minus values...
// * sPm: minus split pressure (P-), Capital script P in reference
// * sMm: minus split Mach Number (M-), Capital script M in reference
// * Reference: "A Sequel to AUSM: AUSM+" Liou, pg 368, Eqns (21a, 21b), 1996
// */
//static void AusmpSplitCalculatorMinus (PetscReal M, PetscReal* sPm, PetscReal* sMm ){
//    if(PetscAbsReal(M) >= 1.0){// Supersonic
//        // sMm:
//        *sMm = sM1m(M);
//
//        // spm:
//        // Equation v1: 1/2*[1 - sign(M)]
//        // Equation v2: 1/2*[1 - |M|/M]
//        *sPm = (*sMm)/(M);
//    }else{// Subsonic
//        {// sMm:
//            // term1 = 1/4*[M-1]^2
//            PetscReal term1 = M - 1.e0;
//            term1 = term1 * term1;
//            term1 = 0.25e+0 * term1;
//            // term2 = [M^2-1]^2
//            PetscReal term2 = M;
//            term2 = term2 * term2;
//            term2 = term2 - 1.e0;
//            term2 = term2 * term2;
//            // Equation: -1/4*[M-1]^2 - beta*[M^2-1]^2
//            *sMm = -term1;//TODO: - AUSMbeta * term2;
//        }
//        {// sPm:
//            // term1 = 1/4*[M-1]^2
//            PetscReal term1 = M - 1.e+0;
//            term1 = term1 * term1;
//            term1 = 0.25e+0 * term1;
//            // term2 = [M^2-1]^2
//            PetscReal term2 = M;
//            term2 = term2 * term2;
//            term2 = term2 - 1.e+0;
//            term2 = term2 * term2;
//            // Equation: 1/4*[M-1]^2*[2+M] - alpha*M*[M^2-1]^2
//            *sPm = term1 * (2.e+0 + M) - AUSMalpha * M * term2;
//        }
//    }
//}
///*
// * Computes the plus values...
// * sPp: plus split pressure (P+), Capital script P in reference
// * sMp: plus split Mach Number (M+), Capital script M in reference
// * Reference: "A Sequel to AUSM: AUSM+" Liou, pg 368, Eqns (21a, 21b), 1996
// */
//static void AusmpSplitCalculatorPlus (PetscReal M, PetscReal* sPp, PetscReal *sMp ){
//    if(PetscAbsReal(M) >= 1.0){// Supersonic
//        // sMp:
//        *sMp = sM1p(M);
//
//        // sPp:
//        // Equation v1: 1/2*[1 + sign(M)]
//        // Equation v2: 1/2*[1 + |M|/M]
//        *sPp = (*sMp)/(M);
//    }else{// Subsonic
//        {// sMp:
//            // term1 = 1/4*[M+1]^2
//            PetscReal term1 = M + 1;
//            term1 = term1 * term1;
//            term1 = 0.25e+0 * term1;
//            // term2 = [M^2-1]^2
//            PetscReal term2 = M;
//            term2 = term2 * term2;
//            term2 = term2 - 1.;
//            term2 = term2 * term2;
//            // Equation: 1/4*[M+1]^2 + beta*[M^2-1]^2
//            *sMp = term1 + AUSMbeta * term2;
//        }
//        {// sPp:
//            // term1 = 1/4*[M+1]^2
//            PetscReal term1 = M + 1.e+0;
//            term1 = term1 * term1;
//            term1 = 0.25e+0 * term1;
//            // term2 = [M^2-1]^2
//            PetscReal term2 = M;
//            term2 = term2 * term2;
//            term2 = term2 - 1.e+0;
//            term2 = term2 * term2;
//            // Equation: 1/4*[M+1]^2*[2-M] + alpha*M*[M^2-1]^2
//            *sPp = (term1 * (2.e+0 - M) + AUSMalpha * M * term2);
//        }
//    }
//}

/* Computes the min/plus values..
 * sPm: minus split pressure (P-), Capital script P in reference
 * sMm: minus split Mach Number (M-), Capital script M in reference
 * sPp: plus split pressure (P+), Capital script P in reference
 * sMp: plus split Mach Number (M+), Capital script M in reference
 */
static void AusmSplitCalculator(PetscReal Mm, PetscReal* sPm, PetscReal* sMm,
                                      PetscReal Mp, PetscReal* sPp, PetscReal *sMp) {

    if (PetscAbsReal(Mm) <= 1. ) {
        *sMm = -0.25 * PetscSqr(Mm - 1);
        *sPm = -(*sMm) * (2 + Mm);
    }else {
        *sMm = 0.5 * (Mm - PetscAbsReal(Mm));
        *sPm = (*sMm) / Mm;
    }
    if (PetscAbsReal(Mp) <= 1. ) {
        *sMp = 0.25 * PetscSqr(Mp + 1);
        *sPp = (*sMp) * (2 - Mp);
    }else {
        *sMp = 0.5 * (Mp + PetscAbsReal(Mp));
        *sPp = (*sMp) / Mp;
    }
}

/**
 * Function to get the density, velocity, and energy from the conserved variables
 * @return
 */
static void DecodeState(PetscInt dim, const PetscReal* conservedValues,  const PetscReal *normal, PetscReal gamma, PetscReal* density,
                                  PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p){

    // decode
    *density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE + dim-1]/(*density);

    // Get the velocity in this direction
    (*velocity) = 0.0;
    for(PetscInt d =0; d < dim; d++){
        (*velocity) += conservedValues[RHOU + d]*normal[d]/(*density);
    }

    // assumed eos
    (*internalEnergy) = (totalEnergy) - 0.5 *(*velocity)*(*velocity);
    *p = (gamma - 1.0)*(*density)*(*internalEnergy);
    *a = PetscSqrtReal(gamma*(*p)/(*density));

    *M = (*velocity)/(*a);
}

static inline void NormVector(PetscInt dim, const PetscReal* in, PetscReal* out){
    PetscReal mag;
    for (PetscInt d=0; d< dim; d++) {
        mag += in[d]*in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d=0; d< dim; d++) {
        out[d] = in[d]/mag;
    }
}

static void ComputeFluxRho(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowParameters* flowParameters = (EulerFlowParameters*)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(area[0]) > 1E-5) {
        // Compute the norm
        PetscReal norm[3];
        NormVector(dim, area, norm);

        // Decode the left and right states
        PetscReal densityL;
        PetscReal velocityL;
        PetscReal internalEnergyL;
        PetscReal aL;
        PetscReal ML;
        PetscReal pL;
        DecodeState(dim, xL, norm, flowParameters->gamma, &densityL, &velocityL, &internalEnergyL, &aL, &ML, &pL);

        PetscReal densityR;
        PetscReal velocityR;
        PetscReal internalEnergyR;
        PetscReal aR;
        PetscReal MR;
        PetscReal pR;
        DecodeState(dim, xR, norm, flowParameters->gamma, &densityR, &velocityR, &internalEnergyR, &aR, &MR, &pR);

        PetscReal sPm;
        PetscReal sPp;
        PetscReal sMm;
        PetscReal sMp;

        AusmSplitCalculator(MR, &sPm, &sMm, ML, &sPp, &sMp);

        // Compute M and P on the face
        PetscReal M = sMm + sMp;

        if(M < 0){
            // M- on Right
            flux[0] = M * densityR * aR * area[0];
        }else{
            // M+ on Left
            flux[0] = M * densityL * aL * area[0];
        }
    }else{
        flux[0] = 0.0;
    }
}

static void ComputeFluxRhoU(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowParameters* flowParameters = (EulerFlowParameters*)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(area[0]) > 1E-5) {
        // Compute the norm
        PetscReal norm[3];
        NormVector(dim, area, norm);

        // Decode the left and right states
        PetscReal densityL;
        PetscReal velocityL;
        PetscReal internalEnergyL;
        PetscReal aL;
        PetscReal ML;
        PetscReal pL;
        DecodeState(dim, xL, norm, flowParameters->gamma, &densityL, &velocityL, &internalEnergyL, &aL, &ML, &pL);

        PetscReal densityR;
        PetscReal velocityR;
        PetscReal internalEnergyR;
        PetscReal aR;
        PetscReal MR;
        PetscReal pR;
        DecodeState(dim, xR, norm, flowParameters->gamma, &densityR, &velocityR, &internalEnergyR, &aR, &MR, &pR);

        PetscReal sPm;
        PetscReal sPp;
        PetscReal sMm;
        PetscReal sMp;

        AusmSplitCalculator(MR, &sPm, &sMm, ML, &sPp, &sMp);

        // Compute M and P on the face
        PetscReal M = sMm + sMp;
        PetscReal p = pR*sPm + pL*sPp;

        if(M < 0){
            // M- on Right
            flux[0] = (M * densityR * aR * velocityR + p) * area[0];;
        }else{
            // M+ on Left
            flux[0] = (M * densityL * aL * velocityL + p) * area[0];;
        }
        flux[1] = 0.0;
    }else{
        flux[0] = 0.0;
        flux[1] = 0.0;
    }
}

static void ComputeFluxRhoE(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *area, const PetscReal *xL, const PetscReal *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowParameters* flowParameters = (EulerFlowParameters*)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(area[0]) > 1E-5) {
        // Compute the norm
        PetscReal norm[3];
        NormVector(dim, area, norm);

        // Decode the left and right states
        PetscReal densityL;
        PetscReal velocityL;
        PetscReal internalEnergyL;
        PetscReal aL;
        PetscReal ML;
        PetscReal pL;
        DecodeState(dim, xL, norm, flowParameters->gamma, &densityL, &velocityL, &internalEnergyL, &aL, &ML, &pL);

        PetscReal densityR;
        PetscReal velocityR;
        PetscReal internalEnergyR;
        PetscReal aR;
        PetscReal MR;
        PetscReal pR;
        DecodeState(dim, xR, norm, flowParameters->gamma, &densityR, &velocityR, &internalEnergyR, &aR, &MR, &pR);

        PetscReal sPm;
        PetscReal sPp;
        PetscReal sMm;
        PetscReal sMp;

        AusmSplitCalculator(MR, &sPm, &sMm, ML, &sPp, &sMp);

        // Compute M and P on the face
        PetscReal M = sMm + sMp;

        if(M < 0){
            // M- on Right
            PetscReal HR = internalEnergyR + velocityR*velocityR/2.0 + pR/densityR;
            flux[0] = (M * densityR * aR * HR) * area[0];;
        }else{
            // M+ on Left
            PetscReal HL = internalEnergyL + velocityL*velocityL/2.0 + pL/densityL;
            flux[0] = (M * densityL * aL * HL) * area[0];;
        }
    }else{
        flux[0] = 0.0;
    }
}

PetscErrorCode CompressibleFlow_SetupFlowParameters(FlowData flowData, EulerFlowParameters* eulerFlowParameters){
    PetscFunctionBeginUser;
    flowData->data = eulerFlowParameters;
    PetscFunctionReturn(0);
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

PetscErrorCode CompressibleFlow_StartProblemSetup(FlowData flowData) {
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    PetscDS prob;
    ierr = DMGetDS(flowData->dm, &prob);CHKERRQ(ierr);

    // Set the flux calculator solver for each component
    ierr = PetscDSSetRiemannSolver(prob, RHO, ComputeFluxRho);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, RHO, flowData);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, RHOU,ComputeFluxRhoU);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, RHOU, flowData);CHKERRQ(ierr);
    ierr = PetscDSSetRiemannSolver(prob, RHOE,ComputeFluxRhoE);CHKERRQ(ierr);
    ierr = PetscDSSetContext(prob, RHOE, flowData);CHKERRQ(ierr);
    ierr = PetscDSSetFromOptions(prob);CHKERRQ(ierr);

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
    EulerFlowParameters* flowParameters = (EulerFlowParameters*)flowData->data;

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
    ierr =  FlowCompleteProblemSetup(flowData, ts);CHKERRQ(ierr);
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