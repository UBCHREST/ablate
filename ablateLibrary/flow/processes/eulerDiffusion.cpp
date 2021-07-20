#include "eulerDiffusion.hpp"
#include <utilities/petscError.hpp>
#include "eulerAdvection.hpp"

PetscErrorCode ablate::flow::processes::EulerDiffusion::UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscScalar *conservedValues,
                                                                                  PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[EulerAdvection::RHO];
    PetscReal totalEnergy = conservedValues[EulerAdvection::RHOE] / density;
    EulerDiffusionData flowParameters = (EulerDiffusionData)ctx;
    PetscErrorCode ierr = flowParameters->computeTemperatureFunction(dim,
                                                                     density,
                                                                     totalEnergy,
                                                                     conservedValues + EulerAdvection::RHOU,
                                                                     flowParameters->numberSpecies ? conservedValues + EulerAdvection::RHOU + dim : NULL,
                                                                     &auxField[T],
                                                                     flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::processes::EulerDiffusion::UpdateAuxVelocityField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscScalar *conservedValues, PetscScalar *auxField,
                                                                               void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[EulerAdvection::RHO];

    for (PetscInt d = 0; d < dim; d++) {
        auxField[d] = conservedValues[EulerAdvection::RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

ablate::flow::processes::EulerDiffusion::EulerDiffusion(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eosIn) : eos(eosIn) {
    PetscNew(&eulerDiffusionData);

    // Store the required data for the low level c functions
    eulerDiffusionData->mu = parameters->Get<PetscReal>("mu", 0.0);
    eulerDiffusionData->k = parameters->Get<PetscReal>("k", 0.0);

    // set the decode state function
    eulerDiffusionData->computeTemperatureFunction = eos->GetComputeTemperatureFunction();
    eulerDiffusionData->computeTemperatureContext = eos->GetComputeTemperatureContext();
    eulerDiffusionData->numberSpecies = eos->GetSpecies().size();
}

ablate::flow::processes::EulerDiffusion::~EulerDiffusion() { PetscFree(eulerDiffusionData); }

void ablate::flow::processes::EulerDiffusion::Initialize(ablate::flow::FVFlow &flow) {
    // if there are any coefficients for diffusion, compute diffusion
    if (eulerDiffusionData->k || eulerDiffusionData->mu) {
        // Register the euler diffusion source terms
        flow.RegisterRHSFunction(CompressibleFlowEulerDiffusion, eulerDiffusionData, "euler", {"euler"}, {"T", "vel"});
    }

    // add in aux update variables TODO: remove hard coded order of the temperature using a aOff type argument
    flow.RegisterAuxFieldUpdate(UpdateAuxTemperatureField, eulerDiffusionData, "T");
    flow.RegisterAuxFieldUpdate(UpdateAuxVelocityField, eulerDiffusionData, "vel");

    // PetscErrorCode PetscOptionsGetBool(PetscOptions options,const char pre[],const char name[],PetscBool *ivalue,PetscBool *set)
    PetscBool automaticTimeStepCalculator = PETSC_TRUE;
    PetscOptionsGetBool(NULL, NULL, "-automaticTimeStepCalculator", &automaticTimeStepCalculator, NULL);
    if (automaticTimeStepCalculator) {
        flow.RegisterComputeTimeStepFunction(ComputeTimeStep, eulerDiffusionData);
    }

    // determine the dim of the problem
    PetscInt dim;
    DMGetDimension(flow.GetDM(), &dim) >> checkError;
    eulerDiffusionData->dtStabilityFactor = (1.0 / 3.0) / dim;
}
PetscErrorCode ablate::flow::processes::EulerDiffusion::CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                                       const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                                       const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                       const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    EulerDiffusionData flowParameters = (EulerDiffusionData)ctx;

    // Compute the stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    ierr = CompressibleFlowComputeStressTensor(dim, flowParameters->mu, gradAuxL + aOff_x[VEL], gradAuxR + aOff_x[VEL], tau);
    CHKERRQ(ierr);

    // for each velocity component
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal viscousFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
            viscousFlux += -fg->normal[d] * tau[c * dim + d];  // This is tau[c][d]
        }

        // add in the contribution
        flux[EulerAdvection::RHOU + c] = viscousFlux;
    }

    // energy equation
    flux[EulerAdvection::RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal heatFlux = 0.0;
        // add in the contributions for this viscous terms
        for (PetscInt c = 0; c < dim; ++c) {
            heatFlux += 0.5 * (auxL[aOff[VEL] + c] + auxR[aOff[VEL] + c]) * tau[d * dim + c];
        }

        // heat conduction (-k dT/dx - k dT/dy - k dT/dz) . n A
        heatFlux += +flowParameters->k * 0.5 * (gradAuxL[aOff_x[T] + d] + gradAuxR[aOff_x[T] + d]);

        // Multiply by the area normal
        heatFlux *= -fg->normal[d];

        flux[EulerAdvection::RHOE] += heatFlux;
    }

    // zero out the density flux
    flux[EulerAdvection::RHO] = 0.0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::processes::EulerDiffusion::CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal *gradVelL, const PetscReal *gradVelR, PetscReal *tau) {
    PetscFunctionBeginUser;
    // pre compute the div of the velocity field
    PetscReal divVel = 0.0;
    for (PetscInt c = 0; c < dim; ++c) {
        divVel += 0.5 * (gradVelL[c * dim + c] + gradVelR[c * dim + c]);
    }

    // March over each velocity component, u, v, w
    for (PetscInt c = 0; c < dim; ++c) {
        // March over each physical coordinate coordinate
        for (PetscInt d = 0; d < dim; ++d) {
            if (d == c) {
                // for the xx, yy, zz, components
                tau[c * dim + d] = 2.0 * mu * (0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) - divVel / 3.0);
            } else {
                // for xy, xz, etc
                tau[c * dim + d] = mu * (0.5 * (gradVelL[c * dim + d] + gradVelR[c * dim + d]) + 0.5 * (gradVelL[d * dim + c] + gradVelR[d * dim + c]));
            }
        }
    }
    PetscFunctionReturn(0);
}

double ablate::flow::processes::EulerDiffusion::ComputeTimeStep(TS ts, ablate::flow::Flow &flow, void *ctx) {
    // Get the dm and current solution vector
    DM dm;
    TSGetDM(ts, &dm) >> checkError;
    Vec v;
    TSGetSolution(ts, &v) >> checkError;

    // Get the flow param
    EulerDiffusionData eulerDiffusionData = (EulerDiffusionData)ctx;

    // Get the fv geom
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(dm, NULL, NULL, &minCellRadius) >> checkError;
    PetscInt cStart, cEnd;
    DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd) >> checkError;
    const PetscScalar *x;
    VecGetArrayRead(v, &x) >> checkError;

    // Get the dim from the dm
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;

    // assume the smallest cell is the limiting factor for now
    const PetscReal dx = 2.0 * minCellRadius;

    // Get field location for euler and densityYi
    auto eulerId = flow.GetFieldId("euler").value();

    // March over each cell
    PetscReal dtMin = 1000.0;
    for (PetscInt c = cStart; c < cEnd; ++c) {
        const PetscReal *xc;
        DMPlexPointGlobalFieldRead(dm, c, eulerId, x, &xc) >> checkError;

        if (xc) {  // must be real cell and not ghost
            PetscReal rho = xc[EulerAdvection::RHO];
            PetscReal nu = eulerDiffusionData->mu / rho;

            PetscReal dt = eulerDiffusionData->dtStabilityFactor * dx / (nu);
            dtMin = PetscMin(dtMin, dt);
        }
    }
    VecRestoreArrayRead(v, &x) >> checkError;
    return dtMin;
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::processes::FlowProcess, ablate::flow::processes::EulerDiffusion, "diffusion for the euler field",
         OPT(ablate::parameters::Parameters, "parameters", "the parameters used by advection"), ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"));
