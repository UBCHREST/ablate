#include "eulerDiffusion.hpp"
#include <utilities/petscError.hpp>
#include "eulerAdvection.hpp"

// When used, you must request euler, then densityYi
PetscErrorCode ablate::flow::processes::EulerDiffusion::UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                  const PetscScalar *conservedValues, PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + EulerAdvection::RHO];
    PetscReal totalEnergy = conservedValues[uOff[0] + EulerAdvection::RHOE] / density;
    EulerDiffusionData flowParameters = (EulerDiffusionData)ctx;
    PetscErrorCode ierr = flowParameters->computeTemperatureFunction(dim,
                                                                     density,
                                                                     totalEnergy,
                                                                     conservedValues + uOff[0] + EulerAdvection::RHOU,
                                                                     flowParameters->numberSpecies ? conservedValues + uOff[1] : NULL,
                                                                     auxField,
                                                                     flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::processes::EulerDiffusion::UpdateAuxVelocityField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], const PetscScalar *conservedValues,
                                                                               PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + EulerAdvection::RHO];

    for (PetscInt d = 0; d < dim; d++) {
        auxField[d] = conservedValues[uOff[0] + EulerAdvection::RHOU + d] / density;
    }

    PetscFunctionReturn(0);
}

ablate::flow::processes::EulerDiffusion::EulerDiffusion(std::shared_ptr<eos::EOS> eosIn, std::shared_ptr<eos::transport::TransportModel> transportModelIn) : eos(eosIn), transportModel(transportModelIn) {
    PetscNew(&eulerDiffusionData);

    // Store the required data for the low level c functions
    if(transportModel) {
        eulerDiffusionData->muFunction = transportModel->GetComputeViscosityFunction();
        eulerDiffusionData->muContext = transportModel->GetComputeViscosityContext();
        eulerDiffusionData->kFunction = transportModel->GetComputeConductivityFunction();
        eulerDiffusionData->kContext = transportModel->GetComputeConductivityContext();
    }else{
        eulerDiffusionData->muFunction = nullptr;
        eulerDiffusionData->muContext =nullptr;
        eulerDiffusionData->kFunction =nullptr;
        eulerDiffusionData->kContext = nullptr;
    }
    // set the decode state function
    eulerDiffusionData->computeTemperatureFunction = eos->GetComputeTemperatureFunction();
    eulerDiffusionData->computeTemperatureContext = eos->GetComputeTemperatureContext();
    eulerDiffusionData->numberSpecies = eos->GetSpecies().size();
}

ablate::flow::processes::EulerDiffusion::~EulerDiffusion() { PetscFree(eulerDiffusionData); }

void ablate::flow::processes::EulerDiffusion::Initialize(ablate::flow::FVFlow &flow) {
    // if there are any coefficients for diffusion, compute diffusion
    if (eulerDiffusionData->kFunction || eulerDiffusionData->muFunction) {
        // Register the euler diffusion source terms
        flow.RegisterRHSFunction(CompressibleFlowEulerDiffusion, eulerDiffusionData, "euler", {"euler"}, {"T", "vel"});
    }

    // If there are species
    if (eulerDiffusionData->numberSpecies > 0) {
        // add in aux update variables
        flow.RegisterAuxFieldUpdate(UpdateAuxVelocityField, eulerDiffusionData, "vel", {"euler"});
        flow.RegisterAuxFieldUpdate(UpdateAuxTemperatureField, eulerDiffusionData, "T", {"euler", "densityYi"});
    } else {
        // add in aux update variables
        flow.RegisterAuxFieldUpdate(UpdateAuxVelocityField, eulerDiffusionData, "vel", {"euler"});
        flow.RegisterAuxFieldUpdate(UpdateAuxTemperatureField, eulerDiffusionData, "T", {"euler"});
    }
}

PetscErrorCode ablate::flow::processes::EulerDiffusion::CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                                       const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                                       const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                       const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int T = 0;
    const int VEL = 1;

    PetscErrorCode ierr;
    EulerDiffusionData flowParameters = (EulerDiffusionData)ctx;

    // Compute mu and k
    PetscReal mu = 0.0;
    flowParameters->muFunction(0.5*(auxL[aOff[0]] + auxR[aOff[0]]), mu, flowParameters->muContext);
    PetscReal k = 0.0;
    flowParameters->kFunction(0.5*(auxL[aOff[0]] + auxR[aOff[0]]), k, flowParameters->kContext);

    // Compute the stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    ierr = CompressibleFlowComputeStressTensor(dim, mu, gradAuxL + aOff_x[VEL], gradAuxR + aOff_x[VEL], tau);
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
        heatFlux += +k * 0.5 * (gradAuxL[aOff_x[T] + d] + gradAuxR[aOff_x[T] + d]);

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

#include "parser/registrar.hpp"
REGISTER(ablate::flow::processes::FlowProcess, ablate::flow::processes::EulerDiffusion, "diffusion for the euler field",
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"), OPT(ablate::eos::transport::TransportModel, "parameters", "the diffusion transport model"));
