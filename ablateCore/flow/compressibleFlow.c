#include "compressibleFlow.h"


static const char *compressibleFlowFieldNames[TOTAL_COMPRESSIBLE_FLOW_FIELDS + 1] = {"density", "momentum", "energy", "unknown"};

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


#define EPS 1.e-6
#define MAXIT 100
PetscErrorCode DetermineStarState(const InitialConditions* setup, StarState* starState){
    // compute the speed of sound
    PetscReal aL = PetscSqrtReal(setup->gamma*setup->pL/setup->rhoL);
    PetscReal aR = PetscSqrtReal(setup->gamma*setup->pR/setup->rhoR);

    //first guess pstar based on two-rarefacation approximation
    starState->pstar = aL+aR - 0.5*(setup->gamma-1.)*(setup->uR-setup->uL);
    starState->pstar  = starState->pstar  / (aL/pow(setup->pL,0.5*(setup->gamma-1.)/setup->gamma) + aR/pow(setup->pR,0.5*(setup->gamma-1.)/setup->gamma) );
    starState->pstar = pow(starState->pstar,2.*setup->gamma/(setup->gamma-1.));
    starState->gamm1 = setup->gamma-1.;
    starState->gamp1 = setup->gamma+1.;

    PressureFunction fL;
    if (starState->pstar <= setup->pL){
        fL = f_and_fprm_rarefaction(starState->pstar, setup->pL,aL,setup->gamma,starState->gamm1,starState->gamp1);
    }else{
        fL = f_and_fprm_shock(starState->pstar,setup->pL,setup->rhoL,setup->gamma,starState->gamm1,starState->gamp1);
    }

    PressureFunction fR;
    if (starState->pstar <= setup->pR) {
        fR = f_and_fprm_rarefaction(starState->pstar, setup->pR, aR, setup->gamma, starState->gamm1, starState->gamp1);
    }else {
        fR = f_and_fprm_shock(starState->pstar, setup->pR, setup->rhoR, setup->gamma, starState->gamm1, starState->gamp1);
    }
    PetscReal delu = setup->uR-setup->uL;

    // iterate using Newton-Rapson
    if ((fL.f+fR.f+delu)> EPS) {
        // iterate using Newton-Rapson
        for(PetscInt it =0; it < MAXIT+4; it++){
            PetscReal pold = starState->pstar;
            starState->pstar = pold - (fL.f+fR.f+delu)/(fL.fprm+fR.fprm);

            if(starState->pstar < 0){
                starState->pstar = EPS;
            }

            if(2.0*PetscAbsReal(starState->pstar - pold)/(starState->pstar + pold) < EPS){
                break;
            }else{
                if(starState->pstar < setup->pL){
                    fL = f_and_fprm_rarefaction(starState->pstar, setup->pL,aL,setup->gamma,starState->gamm1,starState->gamp1);
                }else{
                    fL = f_and_fprm_shock(starState->pstar,setup->pL,setup->rhoL,setup->gamma,starState->gamm1,starState->gamp1);
                }
                if (starState->pstar<=setup->pR) {
                    fR = f_and_fprm_rarefaction(starState->pstar, setup->pR, aR, setup->gamma, starState->gamm1, starState->gamp1);
                }else {
                    fR = f_and_fprm_shock(starState->pstar, setup->pR, setup->rhoR, setup->gamma, starState->gamm1, starState->gamp1);
                }
            }

            if (it>MAXIT){
                SETERRQ(PETSC_COMM_WORLD,1,"error in Riemann.find_pstar - did not converage for pstar" );
            }
        }
    }

    // determine rest of star state
    starState->ustar = 0.5*(setup->uL+setup->uR+fR.f-fL.f);

    // left star state
    PetscReal pratio = starState->pstar/setup->pL;
    if (starState->pstar<=setup->pL) {  // rarefaction
        starState->rhostarL = setup->rhoL * PetscPowReal(pratio, 1. / setup->gamma);
        starState->astarL = aL * PetscPowReal(pratio, 0.5 * starState->gamm1 / setup->gamma);
        starState->SHL = setup->uL - aL;
        starState->STL = starState->ustar - starState->astarL;
    }else {  // #shock
        starState->rhostarL = setup->rhoL * (pratio + starState->gamm1 / starState->gamp1) / (starState->gamm1 * pratio / starState->gamp1 + 1.);
        starState->SL = setup->uL - aL * PetscSqrtReal(0.5 * starState->gamp1 / setup->gamma * pratio + 0.5 * starState->gamm1 / setup->gamma);
    }

    // right star state
    pratio = starState->pstar/setup->pR;
    if (starState->pstar<=setup->pR) {  // # rarefaction
        starState->rhostarR = setup->rhoR * PetscPowReal(pratio, 1. / setup->gamma);
        starState->astarR = aR * PetscPowReal(pratio, 0.5 * starState->gamm1 / setup->gamma);
        starState-> SHR = setup->uR + aR;
        starState->STR = starState->ustar + starState->astarR;
    }else {  // shock
        starState->rhostarR = setup->rhoR * (pratio + starState->gamm1 / starState->gamp1) / (starState->gamm1 * pratio / starState->gamp1 + 1.);
        starState->SR = setup->uR + aR * PetscSqrtReal(0.5 * starState->gamp1 / setup->gamma * pratio + 0.5 * starState->gamm1 / setup->gamma);
    }
    return 0;
}

void SetExactSolutionAtPoint(PetscInt dim, PetscReal xDt, const InitialConditions* setup, const StarState* starState, EulerNode* uu){
    PetscReal p;
    // compute the speed of sound
    PetscReal aL = PetscSqrtReal(setup->gamma*setup->pL/setup->rhoL);
    PetscReal aR = PetscSqrtReal(setup->gamma*setup->pR/setup->rhoR);

    for(PetscInt i =0; i < dim; i++){
        uu->rhoU[i] = 0.0;
    }

    if (xDt <= starState->ustar) {  //# left of contact surface
        if (starState->pstar <= setup->pL) {  // # rarefaction
            if (xDt <= starState->SHL) {
                uu->rho = setup->rhoL;
                p = setup->pL;
                uu->rhoU[0] = setup->uL*uu->rho;
            }else if (xDt <=starState->STL) {  //#SHL < x / t < STL
                PetscReal tmp = 2. / starState->gamp1 + (starState->gamm1 / starState->gamp1 / aL) * (setup->uL - xDt);
                uu->rho = setup->rhoL * pow(tmp, 2. / starState->gamm1);
                uu->rhoU[0] = uu->rho * (2. / starState->gamp1) * (aL + 0.5 * starState->gamm1 * setup->uL + xDt);
                p = setup->pL * pow(tmp, 2. * setup->gamma / starState->gamm1);
            }else {  //# STL < x/t < u*
                uu->rho = starState->rhostarL;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }else{  //# shock
            if (xDt<= starState->SL) {  // # xDt < SL
                uu->rho = setup->rhoL;
                p = setup->pL;
                uu->rhoU[0] = uu->rho * setup->uL;
            }else {  //# SL < xDt < ustar
                uu->rho = starState->rhostarL;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }
    }else{//# right of contact surface
        if (starState->pstar<=setup->pR) {  //# rarefaction
            if (xDt>= starState->SHR) {
                uu->rho = setup->rhoR;
                p = setup->pR;
                uu->rhoU[0] = uu->rho * setup->uR;
            }else if (xDt >= starState->STR) {  // # SHR < x/t < SHR
                PetscReal tmp = 2./starState->gamp1 - (starState->gamm1/starState->gamp1/aR)*(setup->uR-xDt);
                uu->rho = setup->rhoR*PetscPowReal(tmp,2./starState->gamm1);
                uu->rhoU[0] = uu->rho * (2./starState->gamp1)*(-aR + 0.5*starState->gamm1*setup->uR+xDt);
                p = setup->pR*PetscPowReal(tmp,2.*setup->gamma/starState->gamm1);
            }else{ //# u* < x/t < STR
                uu->rho = starState->rhostarR;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }else {//# shock
            if (xDt>= starState->SR) {  // # xDt > SR
                uu->rho = setup->rhoR;
                p = setup->pR;
                uu->rhoU[0] = uu->rho * setup->uR;
            }else {//#ustar < xDt < SR
                uu->rho = starState->rhostarR;
                p = starState->pstar;
                uu->rhoU[0] = uu->rho * starState->ustar;
            }
        }
    }
    PetscReal e =  p/starState->gamm1/uu->rho;
    PetscReal E = e + 0.5*(uu->rhoU[0]/uu->rho)*(uu->rhoU[0]/uu->rho);
    uu->rhoE = uu->rho*E;
}


static void ComputeFluxRho(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const EulerNode *xL, const EulerNode *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowParameters* flowParameters = (EulerFlowParameters*)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(n[0]) > 1E-5) {
        // Setup Godunov
        InitialConditions currentValues;

        currentValues.gamma = flowParameters->gamma;
        currentValues.length = 1.0;

        if (n[0] > 0) {
            currentValues.rhoL = xL->rho;
            currentValues.uL = xL->rhoU[0] / currentValues.rhoL;
            PetscReal eL = (xL->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
            currentValues.pL = (flowParameters->gamma - 1) * currentValues.rhoL * eL;

            currentValues.rhoR = xR->rho;
            currentValues.uR = xR->rhoU[0] / currentValues.rhoR;
            PetscReal eR = (xR->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
            currentValues.pR = (flowParameters->gamma - 1) * currentValues.rhoR * eR;
        }else{
            currentValues.rhoR = xL->rho;
            currentValues.uR = xL->rhoU[0] / currentValues.rhoR;
            PetscReal eR = (xL->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
            currentValues.pR = (flowParameters->gamma - 1) * currentValues.rhoR * eR;

            currentValues.rhoL = xR->rho;
            currentValues.uL = xR->rhoU[0] / currentValues.rhoL;
            PetscReal eL = (xR->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
            currentValues.pL = (flowParameters->gamma - 1) * currentValues.rhoL * eL;
        }
        StarState result;
        DetermineStarState(&currentValues, &result);
        EulerNode exact;
        SetExactSolutionAtPoint(dim, 0.0, &currentValues, &result, &exact);

        PetscReal rho = exact.rho;
        PetscReal u = exact.rhoU[0]/rho;
        PetscReal e = (exact.rhoE / rho) - 0.5 * u * u;
        PetscReal p = (flowParameters->gamma-1) * rho * e;

        flux[0] = rho * u * PetscSignReal(n[0]);
 //       printf("flux qp[%f]: %f  n:%f rL:%f rR:%f\n ", qp[0], flux[0], n[0], currentValues.rhoL, currentValues.rhoR);
//        flux->rhoU[0] = (rho * u * u + p)* PetscSignReal(n[0]);
//        flux->rhoU[1] = 0.0;
//        PetscReal et = e + 0.5 * u * u;
//        flux->rhoE = (rho * u * (et + p / rho))* PetscSignReal(n[0]);

//        printf("%f,%f %f %f,%f\n", qp[0], qp[1], flux[0], n[0], n[1]);mm

    }else{
        flux[0] = 0.0;
//        flux->rhoU[0] =0.0;
//        flux->rhoU[1] = 0.0;
//        flux->rhoE = 0.0;

    }
}

static void ComputeFluxRhoU(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const EulerNode *xL, const EulerNode *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowParameters* flowParameters = (EulerFlowParameters*)flowData->data;
    // this is a hack, only add in flux from left/right
    if(PetscAbs(n[0]) > 1E-5) {
        // Setup Godunov
        InitialConditions currentValues;

        currentValues.gamma = flowParameters->gamma;
        currentValues.length = 1.0;

        if (n[0] > 0) {
            currentValues.rhoL = xL->rho;
            currentValues.uL = xL->rhoU[0] / currentValues.rhoL;
            PetscReal eL = (xL->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
            currentValues.pL = (flowParameters->gamma - 1) * currentValues.rhoL * eL;

            currentValues.rhoR = xR->rho;
            currentValues.uR = xR->rhoU[0] / currentValues.rhoR;
            PetscReal eR = (xR->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
            currentValues.pR = (flowParameters->gamma - 1) * currentValues.rhoR * eR;
        }else{
            currentValues.rhoR = xL->rho;
            currentValues.uR = xL->rhoU[0] / currentValues.rhoR;
            PetscReal eR = (xL->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
            currentValues.pR = (flowParameters->gamma - 1) * currentValues.rhoR * eR;

            currentValues.rhoL = xR->rho;
            currentValues.uL = xR->rhoU[0] / currentValues.rhoL;
            PetscReal eL = (xR->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
            currentValues.pL = (flowParameters->gamma - 1) * currentValues.rhoL * eL;
        }
        StarState result;
        DetermineStarState(&currentValues, &result);
        EulerNode exact;
        SetExactSolutionAtPoint(dim, 0.0, &currentValues, &result, &exact);


        PetscReal rho = exact.rho;
        PetscReal u = exact.rhoU[0]/rho;
        PetscReal e = (exact.rhoE / rho) - 0.5 * u * u;
        PetscReal p = (flowParameters->gamma-1) * rho * e;

//        flux[0] = (rho * u) * PetscSignReal(n[0]);
        flux[0] = (rho * u * u + p)* PetscSignReal(n[0]);
        flux[1] = 0.0;
//        PetscReal et = e + 0.5 * u * u;
//        flux->rhoE = (rho * u * (et + p / rho))* PetscSignReal(n[0]);

//        printf("%f,%f %f %f %f %f\n", qp[0], qp[1], flux->rho, flux->rhoU[0], flux->rhoE, n[0]);

    }else{
        flux[0] = 0.0;
        flux[1] = 0.0;
//        flux->rhoU[0] =0.0;
//        flux->rhoU[1] = 0.0;
//        flux->rhoE = 0.0;

    }
}

static void ComputeFluxRhoE(PetscInt dim, PetscInt Nf, const PetscReal *qp, const PetscReal *n, const EulerNode *xL, const EulerNode *xR, PetscInt numConstants, const PetscScalar constants[], PetscReal *flux, void* ctx) {
    FlowData flowData = (FlowData)ctx;
    EulerFlowParameters* flowParameters = (EulerFlowParameters*)flowData->data;

    // this is a hack, only add in flux from left/right
    if(PetscAbs(n[0]) > 1E-5) {
        // Setup Godunov
        InitialConditions currentValues;

        currentValues.gamma = flowParameters->gamma;
        currentValues.length = 1.0;

        if (n[0] > 0) {
            currentValues.rhoL = xL->rho;
            currentValues.uL = xL->rhoU[0] / currentValues.rhoL;
            PetscReal eL = (xL->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
            currentValues.pL = (flowParameters->gamma - 1) * currentValues.rhoL * eL;

            currentValues.rhoR = xR->rho;
            currentValues.uR = xR->rhoU[0] / currentValues.rhoR;
            PetscReal eR = (xR->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
            currentValues.pR = (flowParameters->gamma - 1) * currentValues.rhoR * eR;
        }else{
            currentValues.rhoR = xL->rho;
            currentValues.uR = xL->rhoU[0] / currentValues.rhoR;
            PetscReal eR = (xL->rhoE / currentValues.rhoR) - 0.5 * currentValues.uR * currentValues.uR;
            currentValues.pR = (flowParameters->gamma - 1) * currentValues.rhoR * eR;

            currentValues.rhoL = xR->rho;
            currentValues.uL = xR->rhoU[0] / currentValues.rhoL;
            PetscReal eL = (xR->rhoE / currentValues.rhoL) - 0.5 * currentValues.uL * currentValues.uL;
            currentValues.pL = (flowParameters->gamma - 1) * currentValues.rhoL * eL;
        }
        StarState result;
        DetermineStarState(&currentValues, &result);
        EulerNode exact;
        SetExactSolutionAtPoint(dim, 0.0, &currentValues, &result, &exact);


        PetscReal rho = exact.rho;
        PetscReal u = exact.rhoU[0]/rho;
        PetscReal e = (exact.rhoE / rho) - 0.5 * u * u;
        PetscReal p = (flowParameters->gamma-1) * rho * e;

//        flux[0] = (rho * u) * PetscSignReal(n[0]);
//        flux[0] = (rho * u * u + p)* PetscSignReal(n[0]);
//        flux[1] = 0.0;
        PetscReal et = e + 0.5 * u * u;
        flux[0] = (rho * u * (et + p / rho))* PetscSignReal(n[0]);

//        printf("%f,%f %f %f %f %f\n", qp[0], qp[1], flux->rho, flux->rhoU[0], flux->rhoE, n[0]);

    }else{
        flux[0] = 0.0;
//        flux[1] = 0.0;
//        flux->rhoU[0] =0.0;
//        flux->rhoU[1] = 0.0;
//        flux->rhoE = 0.0;

    }
}


PetscErrorCode CompressibleFlow_SetupFlowParameters(FlowData flowData, const EulerFlowParameters* eulerFlowParameters){
    flowData->data = eulerFlowParameters;
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
//
//PetscErrorCode IncompressibleFlow_EnableAuxFields(FlowData flowData) {
//    PetscFunctionBeginUser;
//    // Determine the number of dimensions
//    PetscInt dim;
//    PetscErrorCode ierr = DMGetDimension(flowData->dm, &dim);CHKERRQ(ierr);
//
//    ierr = FlowRegisterAuxField(flowData, incompressibleSourceFieldNames[MOM] , "momentum_source_", dim);CHKERRQ(ierr);
//    ierr = FlowRegisterAuxField(flowData, incompressibleSourceFieldNames[MASS] , "mass_source_", 1);CHKERRQ(ierr);
//    ierr = FlowRegisterAuxField(flowData, incompressibleSourceFieldNames[ENERGY] , "energy_source_", 1);CHKERRQ(ierr);
//    PetscFunctionReturn(0);
//}
//
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

static PetscErrorCode  ComputeTimeStep(TS ts){
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
    MPI_Comm_rank(PetscObjectComm(ts), &rank);
    printf("dtMin(%d): %f\n", rank,dtMin );

    PetscReal dtMinGlobal;
    ierr = MPI_Allreduce(&dtMin, &dtMinGlobal, 1,MPIU_REAL, MPI_MIN, PetscObjectComm(ts));

    PetscPrintf(PetscObjectComm(ts), "TimeStep: %f\n", dtMinGlobal);
    ierr = TSSetTimeStep(ts, dtMinGlobal);CHKERRQ(ierr);

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