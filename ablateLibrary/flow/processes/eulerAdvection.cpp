#include "eulerAdvection.hpp"
#include <utilities/petscError.hpp>
#include "flow/fluxDifferencer/ausmFluxDifferencer.hpp"

typedef enum { RHO, RHOE, RHOU, RHOV, RHOW, TOTAL_COMPRESSIBLE_FLOW_COMPONENTS } CompressibleFlowComponents;

static inline void NormVector(PetscInt dim, const PetscReal* in, PetscReal* out) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    mag = PetscSqrtReal(mag);
    for (PetscInt d = 0; d < dim; d++) {
        out[d] = in[d] / mag;
    }
}

static inline PetscReal MagVector(PetscInt dim, const PetscReal* in) {
    PetscReal mag = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        mag += in[d] * in[d];
    }
    return PetscSqrtReal(mag);
}

/**
 * Function to get the density, velocity, and energy from the conserved variables
 * @return
 */
static void DecodeEulerState(ablate::flow::processes::EulerAdvection::EulerAdvectionData flowData, PetscInt dim, const PetscReal* conservedValues, const PetscReal* densityYi, const PetscReal* normal,
                             PetscReal* density, PetscReal* normalVelocity, PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a, PetscReal* M, PetscReal* p) {
    // decode
    *density = conservedValues[RHO];
    PetscReal totalEnergy = conservedValues[RHOE] / (*density);

    // Get the velocity in this direction
    (*normalVelocity) = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[RHOU + d] / (*density);
        (*normalVelocity) += velocity[d] * normal[d];
    }

    // decode the state in the eos
    flowData->decodeStateFunction(dim, *density, totalEnergy, velocity, densityYi, internalEnergy, a, p, flowData->decodeStateFunctionContext);
    *M = (*normalVelocity) / (*a);
}

PetscErrorCode ablate::flow::processes::EulerAdvection::CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x,
                                                                                         const PetscScalar* fieldL, const PetscScalar* fieldR, const PetscScalar* gradL, const PetscScalar* gradR,
                                                                                         const PetscInt* aOff, const PetscInt* aOff_x, const PetscScalar* auxL, const PetscScalar* auxR,
                                                                                         const PetscScalar* gradAuxL, const PetscScalar* gradAuxR, PetscScalar* flux, void* ctx) {
    EulerAdvectionData eulerAdvectionData = (EulerAdvectionData)ctx;
    PetscFunctionBeginUser;

    const int EULER_FIELD = 0;
    const int YI_FIELD = 1;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;
    const PetscReal* densityYiL = eulerAdvectionData->numberSpecies > 0 ? fieldL + uOff[YI_FIELD] : NULL;
    DecodeEulerState(eulerAdvectionData, dim, fieldL + uOff[EULER_FIELD], densityYiL, norm, &densityL, &normalVelocityL, velocityL, &internalEnergyL, &aL, &ML, &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    const PetscReal* densityYiR = eulerAdvectionData->numberSpecies > 0 ? fieldR + uOff[YI_FIELD] : NULL;
    DecodeEulerState(eulerAdvectionData, dim, fieldR + uOff[EULER_FIELD], densityYiR, norm, &densityR, &normalVelocityR, velocityR, &internalEnergyR, &aR, &MR, &pR);

    // get the face values
    PetscReal massFlux;
    PetscReal p12;

    /*void (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
        PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
        PetscReal * m12, PetscReal *p12);*/
    eulerAdvectionData->fluxDifferencer(eulerAdvectionData->fluxDifferencerCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, &p12);

    if(massFlux > 0){
        flux[RHO] = massFlux * areaMag;
        PetscReal velMagL = MagVector(dim, velocityL);
        PetscReal HL = internalEnergyL + velMagL * velMagL / 2.0 + pL / densityL;
        flux[RHOE] = HL * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[RHOU + n] = velocityL[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
    }else{
        flux[RHO] = massFlux * areaMag;
        PetscReal velMagR = MagVector(dim, velocityR);
        PetscReal HR = internalEnergyR + velMagR * velMagR / 2.0 + pR / densityR;
        flux[RHOE] = HR * massFlux * areaMag;
        for (PetscInt n = 0; n < dim; n++) {
            flux[RHOU + n] = velocityR[n] * massFlux * areaMag + p12 * fg->normal[n];
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::processes::EulerAdvection::CompressibleFlowSpeciesAdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x,
                                                                                             const PetscScalar* fieldL, const PetscScalar* fieldR, const PetscScalar* gradL, const PetscScalar* gradR,
                                                                                             const PetscInt* aOff, const PetscInt* aOff_x, const PetscScalar* auxL, const PetscScalar* auxR,
                                                                                             const PetscScalar* gradAuxL, const PetscScalar* gradAuxR, PetscScalar* flux, void* ctx) {
    EulerAdvectionData eulerAdvectionData = (EulerAdvectionData)ctx;
    PetscFunctionBeginUser;

    // Compute the norm
    PetscReal norm[3];
    NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = MagVector(dim, fg->normal);

    const int EULER_FIELD = 0;
    const int YI_FIELD = 1;

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    PetscReal internalEnergyL;
    PetscReal aL;
    PetscReal ML;
    PetscReal pL;
    DecodeEulerState(eulerAdvectionData, dim, fieldL + uOff[EULER_FIELD], fieldL + uOff[YI_FIELD], norm, &densityL, &normalVelocityL, velocityL, &internalEnergyL, &aL, &ML, &pL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    PetscReal internalEnergyR;
    PetscReal aR;
    PetscReal MR;
    PetscReal pR;
    DecodeEulerState(eulerAdvectionData, dim, fieldR + uOff[EULER_FIELD], fieldR + uOff[YI_FIELD], norm, &densityR, &normalVelocityR, velocityR, &internalEnergyR, &aR, &MR, &pR);

    // get the face values
    PetscReal massFlux;

    /*void (*)(void* ctx, PetscReal uL, PetscReal aL, PetscReal rhoL, PetscReal pL,
    PetscReal uR, PetscReal aR, PetscReal rhoR, PetscReal pR,
    PetscReal * m12, PetscReal *p12);*/
    eulerAdvectionData->fluxDifferencer(eulerAdvectionData->fluxDifferencerCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, NULL);
    if(massFlux > 0){
        // march over each gas species
        for (PetscInt sp = 0; sp < eulerAdvectionData->numberSpecies; sp++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[sp] = (massFlux * fieldL[uOff[YI_FIELD] + sp]) * areaMag;
        }
    }else{
        // march over each gas species
        for (PetscInt sp = 0; sp < eulerAdvectionData->numberSpecies; sp++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[sp] = (massFlux * fieldR[uOff[YI_FIELD] + sp]) * areaMag;
        }
    }


    PetscFunctionReturn(0);
}

ablate::flow::processes::EulerAdvection::EulerAdvection(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eosIn,
                                                        std::shared_ptr<fluxDifferencer::FluxDifferencer> fluxDifferencerIn)
    : eos(eosIn), fluxDifferencer(fluxDifferencerIn == nullptr ? std::make_shared<fluxDifferencer::AusmFluxDifferencer>() : fluxDifferencerIn) {
    PetscNew(&eulerAdvectionData);

    // Store the required data for the low level c functions
    eulerAdvectionData->cfl = parameters->Get<PetscReal>("cfl", 0.5);

    // set the decode state function
    eulerAdvectionData->decodeStateFunction = eos->GetDecodeStateFunction();
    eulerAdvectionData->decodeStateFunctionContext = eos->GetDecodeStateContext();
    eulerAdvectionData->numberSpecies = eos->GetSpecies().size();

    // extract the difference function from fluxDifferencer object
    eulerAdvectionData->fluxDifferencer = fluxDifferencer->GetFluxDifferencerFunction();
}

ablate::flow::processes::EulerAdvection::~EulerAdvection() { PetscFree(eulerAdvectionData); }

void ablate::flow::processes::EulerAdvection::Initialize(ablate::flow::FVFlow& flow) {
    // Register the euler source terms
    if (eos->GetSpecies().empty()) {
        flow.RegisterRHSFunction(CompressibleFlowComputeEulerFlux, eulerAdvectionData, "euler", {"euler"}, {});
    } else {
        flow.RegisterRHSFunction(CompressibleFlowComputeEulerFlux, eulerAdvectionData, "euler", {"euler", "densityYi"}, {});
        flow.RegisterRHSFunction(CompressibleFlowSpeciesAdvectionFlux, eulerAdvectionData, "densityYi", {"euler", "densityYi"}, {});
    }

    // PetscErrorCode PetscOptionsGetBool(PetscOptions options,const char pre[],const char name[],PetscBool *ivalue,PetscBool *set)
    PetscBool automaticTimeStepCalculator = PETSC_TRUE;
    PetscOptionsGetBool(NULL, NULL, "-automaticTimeStepCalculator", &automaticTimeStepCalculator, NULL);
    if (automaticTimeStepCalculator) {
        flow.RegisterComputeTimeStepFunction(ComputeTimeStep, eulerAdvectionData);
    }
}

double ablate::flow::processes::EulerAdvection::ComputeTimeStep(TS ts, ablate::flow::Flow& flow, void* ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    Vec v;
    TSGetSolution(ts, &v) >> checkError;

    // Get the flow param
    EulerAdvectionData eulerAdvectionData = (EulerAdvectionData)ctx;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> checkError;
    PetscInt cStart, cEnd;
    DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd) >> checkError;
    const PetscScalar* x;
    VecGetArrayRead(v, &x) >> checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 * minCellRadius;

    // Get field location for euler and densityYi
    auto eulerId = flow.GetFieldId("euler").value();
    auto densityYiId = flow.GetFieldId("densityYi").value_or(-1);

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscReal* xc;
        const PetscReal* densityYi = NULL;
        DMPlexPointGlobalFieldRead(dm, c, eulerId, x, &xc) >> checkError;

        if (densityYiId >= 0) {
            DMPlexPointGlobalFieldRead(dm, c, densityYiId, x, &densityYi) >> checkError;
        }

        if (xc) {  // must be real cell and not ghost
            PetscReal rho = xc[RHO];
            PetscReal vel[3];
            for (PetscInt i = 0; i < dim; i++) {
                vel[i] = xc[RHOU + i] / rho;
            }

            // Get the speed of sound from the eos
            PetscReal ie;
            PetscReal a;
            PetscReal p;
            eulerAdvectionData->decodeStateFunction(dim, rho, xc[RHOE] / rho, vel, densityYi, &ie, &a, &p, eulerAdvectionData->decodeStateFunctionContext) >> checkError;

            PetscReal u = xc[RHOU] / rho;
            PetscReal dt = eulerAdvectionData->cfl * dx / (a + PetscAbsReal(u));
            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> checkError;
    return dtMin;
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::processes::FlowProcess, ablate::flow::processes::EulerAdvection, "build advection for the euler field and species",
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection"), ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"),
         OPT(ablate::flow::fluxDifferencer::FluxDifferencer, "fluxDifferencer", "the flux differencer (defaults to AUSM)"));
