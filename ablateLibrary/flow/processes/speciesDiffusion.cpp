#include "speciesDiffusion.hpp"
#include "eulerAdvection.hpp"

ablate::flow::processes::SpeciesDiffusion::SpeciesDiffusion(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eos) {
    PetscNew(&speciesDiffusionData);
    speciesDiffusionData->diff = parameters->Get<PetscReal>("D", 0.0);
    speciesDiffusionData->numberSpecies = eos->GetSpecies().size();
}
ablate::flow::processes::SpeciesDiffusion::~SpeciesDiffusion() { PetscFree(speciesDiffusionData); }

void ablate::flow::processes::SpeciesDiffusion::Initialize(ablate::flow::FVFlow &flow) {
    // if there are any coefficients for diffusion, compute diffusion
    if (speciesDiffusionData->numberSpecies > 0) {
        if (speciesDiffusionData->diff > 0) {
            // Register the euler diffusion source terms
            flow.RegisterRHSFunction(SpeciesDiffusionEnergyFlux, speciesDiffusionData, "euler", {"euler"}, {"yi"});
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
                                                                                     const PetscScalar *gradAuxR, PetscScalar *fL, void *ctx) {
    return 0;
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

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            // speciesFlux(-rho Di dYi/dx - rho Di dYi/dy - rho Di dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal speciesFlux = -fg->normal[d] * density * flowParameters->diff * 0.5 * (gradAuxL[offset] + gradAuxR[offset]);
            flux[sp] += speciesFlux;
        }
    }

    PetscFunctionReturn(0);
}
