#include "LES.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::processes::LES::LES(std::shared_ptr<eos::transport::TransportModel> transportModelIn)
   : transportModel(std::move(transportModelIn)){}

//?? I have registered separate sources for momentum and energy fluxes. any problem??//
void ablate::finiteVolume::processes::LES::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {

    if (transportModel) {
        if (diffusionData.diffFunction.function) {
            // Register the euler/Momentum LESdiffusion source term
            flow.RegisterRHSFunction(LESMomentumFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::EULER_FIELD,
                                     {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_EV_FIELD},
                                     {CompressibleFlowFields::VELOCITY_FIELD});
            // Register the euler/Energy LESdiffusion source term
            flow.RegisterRHSFunction(LESEnergyFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::EULER_FIELD,
                                     {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_EV_FIELD},
                                     {CompressibleFlowFields::TEMPERATURE_FIELD, CompressibleFlowFields::VELOCITY_FIELD});
            // Register the Species LESdiffusion source term
            flow.RegisterRHSFunction(LESSpeciesFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::DENSITY_YI_FIELD,
                                     {CompressibleFlowFields::DENSITY_YI_FIELD, CompressibleFlowFields::DENSITY_EV_FIELD},
                                     {CompressibleFlowFields::EV_FIELD});
            // Register the ev LESdiffusion source term
            flow.RegisterRHSFunction(LESevFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::EV_FIELD,
                                     {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_EV_FIELD},
                                     {CompressibleFlowFields::EV_FIELD});
        }
    }

    diffusionData.computeTemperatureFunction = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, flow.GetSubDomain().GetFields());
}
/*?? I use velocity and temperature in two of my functions, LESEnergyFlux and CompressibleFlowComputeLESStressTensor, should I compute those fields using "UpdateAux.." function or those fields get
updated by navierStokesTransport ??*/
PetscErrorCode ablate::finiteVolume::processes::LES::LESMomentumFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x, const PetscScalar* field,
                                                                     const PetscScalar* grad, const PetscInt* aOff, const PetscInt* aOff_x, const PetscScalar* aux, const PetscScalar* gradAux,
                                                                     PetscScalar* flux, void* ctx) {
    PetscFunctionBeginUser;
    const int VEL = 1;

    for (PetscInt d = 0; d < dim; d++) {
        flux[CompressibleFlowFields::RHOU + d] = 0.0;
    }
    // Compute the LES stress tensor tau
    PetscReal LESTau[9];  // Maximum size without symmetry

    PetscErrorCode ierr;
    ierr = CompressibleFlowComputeLESStressTensor(dim, fg, gradAux + aOff_x[VEL], uOff_x, field, ctx, LESTau);
    CHKERRQ(ierr);

    // for each velocity component
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal LESViscousFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
            LESViscousFlux += -fg->normal[d] * 2 * LESTau[c * dim + d];  // This is LESTau[c][d]
        }

        // add in the contribution
        flux[CompressibleFlowFields::RHOU + c] = LESViscousFlux;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LESEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                   const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                   PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;

    const int T = 0;
    const int VEL = 1;

    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    // compute the temperature in this volume
    PetscReal temperature;
    ierr = flowParameters->computeTemperatureFunction.function(field, &temperature, flowParameters->computeTemperatureFunction.context.get());
    CHKERRQ(ierr);

    // set the fluxes to zero
    flux[CompressibleFlowFields::RHOE] = 0.0;

    // Compute the les stress tensor LESTau
    PetscReal LESTau[9];  // Maximum size without symmetry
    PetscReal mut;

    ierr = LESViscosity(dim, fg, field, uOff, ctx, mut);
    CHKERRQ(ierr);
    ierr = CompressibleFlowComputeLESStressTensor(dim, fg, gradAux + aOff_x[VEL], uOff_x, field, ctx, LESTau);
    CHKERRQ(ierr);

    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal LESHeatFlux = 0.0;
        // add in the contributions for this viscous terms
        for (PetscInt c = 0; c < dim; ++c) {
            LESHeatFlux += aux[aOff[VEL] + c] * LESTau[d * dim + c];
        }

        // LES heat conduction (-kt dT/dx - kt dT/dy - kt  dT/dz) . n A
        LESHeatFlux += -c_p * mut * gradAux[aOff_x[T] + d];

        // Multiply by the area normal
        LESHeatFlux *= -fg->normal[d];

        flux[CompressibleFlowFields::RHOE] += LESHeatFlux;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LESSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                    const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                    PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;

    // this order is based upon the order that they are passed into RegisterRHSFunction
    PetscReal mut;
    const int yi = 0;

    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    ierr = LESViscosity(dim, fg, field, uOff, ctx, mut);
    CHKERRQ(ierr);

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            // LESspeciesFlux(-rho mut dYi/dx - mut dYi/dy - rho mut dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal LESspeciesFlux = fg->normal[d] * mut * gradAux[offset];
            flux[sp] += LESspeciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LESevFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                               const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                               PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int euler = 0;
    const int VEL = 1;

    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;
    const PetscFVCellGeom* cg;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // Compute the LES stress tensor tau
    PetscReal LESTau[9];  // Maximum size without symmetry
    const int EV_FIELD = 0;
    PetscReal mut;

    ierr = LESViscosity(dim, fg, field, uOff, ctx, mut);
    CHKERRQ(ierr);
    ierr = CompressibleFlowComputeLESStressTensor(dim, fg, gradAux + aOff_x[VEL], uOff, field, ctx, LESTau);
    CHKERRQ(ierr);

    // energy equation
    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        flux[ev] = 0.0;

        for (PetscInt d = 0; d < dim; ++d) {
            PetscReal LESevFlux = 0.0;
            for (PetscInt c = 0; c < dim; ++c) {
                //
                LESevFlux += -cg->volume * density * LESTau[d * dim + c] * LESTau[d * dim + c] / mut / 2.0;
            }

            //  LESevFlux( rho Di dEVi/dx + rho Di dEVi/dy + rho Di dEVi//dz) . n A +  LESevFlux(-rho ce EV^3/2 ) . n A
            const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;
            LESevFlux += -fg->normal[d] * density * (-mut * gradAux[offset] + c_e * field[uOff[EV_FIELD] + ev] * sqrt(field[uOff[EV_FIELD] + ev]));

            flux[ev] += LESevFlux;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::CompressibleFlowComputeLESStressTensor(PetscInt dim, const PetscFVFaceGeom* fg, const PetscReal* gradVel, const PetscInt uOff[],
                                                                                            const PetscScalar field[], void* ctx, PetscReal* LESTau) {
    PetscFunctionBeginUser;
    // pre-compute the div of the velocity field
    PetscReal divVel = 0.0;
    PetscReal mut;

    PetscErrorCode ierr;
    ierr = LESViscosity(dim, fg, field, uOff, ctx, mut);
    CHKERRQ(ierr);

    for (PetscInt c = 0; c < dim; ++c) {
        divVel += gradVel[c * dim + c];
    }

    // March over each velocity component, u, v, w
    for (PetscInt c = 0; c < dim; ++c) {
        // March over each physical coordinates
        for (PetscInt d = 0; d < dim; ++d) {
            if (d == c) {
                // for the xx, yy, zz, components
                LESTau[c * dim + d] = -2.0 * mut * ((gradVel[c * dim + d]) - divVel / 3.0);
            } else {
                // for xy, xz, etc
                LESTau[c * dim + d] = -2.0 * mut * ((gradVel[c * dim + d]) + (gradVel[d * dim + c]));
            }
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LESViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar field[], const PetscInt uOff[], void* ctx, PetscReal& mut) {
    PetscFunctionBeginUser;
    const int euler = 0;
    const int EV_FIELD = 0;

    auto flowParameters = (DiffusionData*)ctx;

    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // get the current ev from ev_field
    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        // compute turbulent kinetic energy
        const PetscReal tke = field[uOff[EV_FIELD] + ev];
        // compute LES viscosity
        mut += c_k * density * sqrt(areaMag * tke);
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::LES, "Creating LES sources for Navier-Stokes Eqs.",
         OPT(ablate::eos::transport::TransportModel, "transport", "the diffusion transport model (default is no diffusion)"))
