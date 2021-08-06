#include "speciesDiffusion.hpp"
#include "eulerAdvection.hpp"

ablate::flow::processes::SpeciesDiffusion::SpeciesDiffusion(std::shared_ptr<eos::EOS> eosIn,  std::shared_ptr<eos::transport::TransportModel> transportModelIn): eos(eosIn), transportModel(transportModelIn) {
    PetscNew(&speciesDiffusionData);

    if(transportModel) {
        speciesDiffusionData->diffFunction = transportModel->GetComputeDiffusivityFunction();
        speciesDiffusionData->diffContext = transportModel->GetComputeDiffusivityContext();
    }else{
        speciesDiffusionData->diffFunction  = nullptr;
        speciesDiffusionData->diffContext =nullptr;
    }

    // set the eos functions
    speciesDiffusionData->numberSpecies = eos->GetSpecies().size();
    speciesDiffusionData->computeTemperatureFunction = eos->GetComputeTemperatureFunction();
    speciesDiffusionData->computeTemperatureContext = eos->GetComputeTemperatureContext();

    speciesDiffusionData->computeSpeciesSensibleEnthalpyFunction = eos->GetComputeSpeciesSensibleEnthalpyFunction();
    speciesDiffusionData->computeSpeciesSensibleEnthalpyContext = eos->GetComputeSpeciesSensibleEnthalpyContext();
    speciesDiffusionData->speciesSpeciesSensibleEnthalpy.resize(speciesDiffusionData->numberSpecies);
}
ablate::flow::processes::SpeciesDiffusion::~SpeciesDiffusion() { PetscFree(speciesDiffusionData); }

void ablate::flow::processes::SpeciesDiffusion::Initialize(ablate::flow::FVFlow &flow) {
    // if there are any coefficients for diffusion, compute diffusion
    if (speciesDiffusionData->numberSpecies > 0) {
        if (speciesDiffusionData->diffFunction) {
            // Register the euler diffusion source terms
            flow.RegisterRHSFunction(SpeciesDiffusionEnergyFlux, speciesDiffusionData, "euler", {"euler", "densityYi"}, {"yi"});
            flow.RegisterRHSFunction(SpeciesDiffusionSpeciesFlux, speciesDiffusionData, "densityYi", {"euler"}, {"yi"});
        }

        flow.RegisterAuxFieldUpdate(UpdateAuxMassFractionField, speciesDiffusionData, "yi", {"euler", "densityYi"});
    }
}

PetscErrorCode ablate::flow::processes::SpeciesDiffusion::UpdateAuxMassFractionField(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[],
                                                                                     const PetscScalar *conservedValues, PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + EulerAdvection::RHO];

    SpeciesDiffusionData flowParameters = (SpeciesDiffusionData)ctx;

    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; sp++) {
        auxField[sp] = conservedValues[uOff[1] + sp] / density;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::flow::processes::SpeciesDiffusion::SpeciesDiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                                     const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                                     const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                     const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;

    SpeciesDiffusionData flowParameters = (SpeciesDiffusionData)ctx;

    // get the current density from euler
    const PetscReal density = 0.5 * (fieldL[uOff[euler] + EulerAdvection::RHO] + fieldR[uOff[euler] + EulerAdvection::RHO]);

    // compute the temperature in this volume
    PetscErrorCode ierr;
    PetscReal temperatureLeft;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldL[uOff[euler] + EulerAdvection::RHO],
                                                      fieldL[uOff[euler] + EulerAdvection::RHOE] / fieldL[uOff[euler] + EulerAdvection::RHO],
                                                      fieldL + uOff[euler] + EulerAdvection::RHOU,
                                                      auxL + aOff[yi],
                                                      &temperatureLeft,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscReal temperatureRight;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldR[uOff[euler] + EulerAdvection::RHO],
                                                      fieldR[uOff[euler] + EulerAdvection::RHOE] / fieldR[uOff[euler] + EulerAdvection::RHO],
                                                      fieldR + uOff[euler] + EulerAdvection::RHOU,
                                                      auxR + aOff[yi],
                                                      &temperatureRight,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    // compute the enthalpy for each species
    PetscReal temperature = 0.5 * (temperatureLeft + temperatureRight);
    flowParameters->computeSpeciesSensibleEnthalpyFunction(temperature, &flowParameters->speciesSpeciesSensibleEnthalpy[0], flowParameters->computeSpeciesSensibleEnthalpyContext);

    // set the non rho E fluxes to zero
    flux[EulerAdvection::RHO] = 0.0;
    flux[EulerAdvection::RHOE] = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        flux[EulerAdvection::RHOU + d] = 0.0;
    }

    // compute diff
    PetscReal diffLeft = 0.0;
    flowParameters->diffFunction(temperatureLeft,fieldL[uOff[euler] + EulerAdvection::RHO], auxL + aOff[yi], diffLeft, flowParameters->diffContext );
    PetscReal diffRight = 0.0;
    flowParameters->diffFunction(temperatureRight,fieldR[uOff[euler] + EulerAdvection::RHO], auxR + aOff[yi], diffRight, flowParameters->diffContext );
    PetscReal diff = 0.5*(diffLeft + diffRight);

    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * diff * flowParameters->speciesSpeciesSensibleEnthalpy[sp] * 0.5 * (gradAuxL[offset] + gradAuxR[offset]);
            flux[EulerAdvection::RHOE] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::flow::processes::SpeciesDiffusion::SpeciesDiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                                      const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff,
                                                                                      const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL,
                                                                                      const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int yi = 0;
    const int euler = 0;

    SpeciesDiffusionData flowParameters = (SpeciesDiffusionData)ctx;

    // get the current density from euler
    const PetscReal density = 0.5 * (fieldL[uOff[euler] + EulerAdvection::RHO] + fieldR[uOff[euler] + EulerAdvection::RHO]);

    PetscErrorCode ierr;
    PetscReal temperatureLeft;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldL[uOff[euler] + EulerAdvection::RHO],
                                                      fieldL[uOff[euler] + EulerAdvection::RHOE] / fieldL[uOff[euler] + EulerAdvection::RHO],
                                                      fieldL + uOff[euler] + EulerAdvection::RHOU,
                                                      auxL + aOff[yi],
                                                      &temperatureLeft,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    PetscReal temperatureRight;
    ierr = flowParameters->computeTemperatureFunction(dim,
                                                      fieldR[uOff[euler] + EulerAdvection::RHO],
                                                      fieldR[uOff[euler] + EulerAdvection::RHOE] / fieldR[uOff[euler] + EulerAdvection::RHO],
                                                      fieldR + uOff[euler] + EulerAdvection::RHOU,
                                                      auxR + aOff[yi],
                                                      &temperatureRight,
                                                      flowParameters->computeTemperatureContext);
    CHKERRQ(ierr);

    // compute diff
    PetscReal diffLeft = 0.0;
    flowParameters->diffFunction(temperatureLeft,fieldL[uOff[euler] + EulerAdvection::RHO], auxL + aOff[yi], diffLeft, flowParameters->diffContext );
    PetscReal diffRight = 0.0;
    flowParameters->diffFunction(temperatureRight,fieldR[uOff[euler] + EulerAdvection::RHO], auxR + aOff[yi], diffRight, flowParameters->diffContext );
    PetscReal diff = 0.5*(diffLeft + diffRight);

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * diff * 0.5 * (gradAuxL[offset] + gradAuxR[offset]);
            flux[sp] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::processes::FlowProcess, ablate::flow::processes::SpeciesDiffusion, "diffusion for the species yi field",
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"), OPT(ablate::eos::transport::TransportModel, "parameters", "the diffusion transport model"));
