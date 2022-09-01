#include "les.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::processes::LES::LES( std::shared_ptr<eos::EOS> eosIn) : eos(std::move(eosIn)) {
    // diffusionData.numberSpecies = (PetscInt)eos->GetSpecies().size();
}

void ablate::finiteVolume::processes::LES::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_EV_FIELD)) {
        auto conservedForm = flow.GetSubDomain().GetField(CompressibleFlowFields::DENSITY_EV_FIELD);
        diffusionData.numberEV = conservedForm.numberComponents;


        if (!flow.GetSubDomain().ContainsField(CompressibleFlowFields::EV_FIELD)) {
            throw std::invalid_argument("The ablate::finiteVolume::processes::EVTransport process expects the conserved (" + CompressibleFlowFields::DENSITY_EV_FIELD + ") and non-conserved (" + CompressibleFlowFields::EV_FIELD +
                                        ") extra variables to be in the flow.");
        }
        /*
        const auto& densityEv = flow.GetSubDomain().GetField("densityEv");
        const auto& extraVariableList = densityEv.components;

        diffusionData.tke_ev = -1;
        for (std::size_t ev = 0; ev < extraVariableList.size(); ev++) {
            if (extraVariableList[ev] == tke) {
                diffusionData.tke_ev = ev;
            }
        }
        if ( diffusionData.tke_ev < 0) {
            throw std::invalid_argument("The LES solver cannot find the " " tke");
        }
*/

        flow.RegisterRHSFunction(
            LesMomentumFlux, &diffusionData, CompressibleFlowFields::EULER_FIELD, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::VELOCITY_FIELD, CompressibleFlowFields::EV_FIELD});
        // Register the euler/Energy LESdiffusion source termtke
        flow.RegisterRHSFunction(LesEnergyFlux,
                                 &diffusionData,
                                 CompressibleFlowFields::EULER_FIELD,
                                 {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_EV_FIELD},
                                 {CompressibleFlowFields::TEMPERATURE_FIELD, CompressibleFlowFields::VELOCITY_FIELD, CompressibleFlowFields::EV_FIELD});
        // Register the Species LESdiffusion source term
        if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_YI_FIELD)) {
            flow.RegisterRHSFunction(LesSpeciesFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::DENSITY_YI_FIELD,
                                     {CompressibleFlowFields::DENSITY_YI_FIELD, CompressibleFlowFields::DENSITY_EV_FIELD},
                                     {CompressibleFlowFields::EV_FIELD, CompressibleFlowFields::VELOCITY_FIELD, CompressibleFlowFields::YI_FIELD});

            // Register the ev LESdiffusion source term
            flow.RegisterRHSFunction(LesEvFlux,
                                     &diffusionData,
                                     CompressibleFlowFields::EV_FIELD,
                                     {CompressibleFlowFields::EULER_FIELD, CompressibleFlowFields::DENSITY_EV_FIELD},
                                     {CompressibleFlowFields::VELOCITY_FIELD, CompressibleFlowFields::EV_FIELD});
        }


    }
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesMomentumFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x, const PetscScalar* field,
                                                                     const PetscScalar* grad, const PetscInt* aOff, const PetscInt* aOff_x, const PetscScalar* aux, const PetscScalar* gradAux,
                                                                     PetscScalar* flux, void* ctx) {
    PetscFunctionBeginUser;
    const int VEL = 1;

    for (PetscInt d = 0; d < dim; d++) {
        flux[CompressibleFlowFields::RHOU + d] = 0.0;
    }
    // Compute the LES stress tensor tau
    PetscReal lestau[9];  // Maximum size without symmetry

    PetscErrorCode ierr;
    ierr = CompressibleFlowComputeLesStressTensor(dim, fg, gradAux + aOff_x[VEL], uOff_x, field, ctx, lestau);
    CHKERRQ(ierr);

    // for each velocity component
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal lesViscousFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
            lesViscousFlux += -fg->normal[d] * 2 * lestau[c * dim + d];  // This is lesTau[c][d]
        }

        // add in the contribution
        flux[CompressibleFlowFields::RHOU + c] = lesViscousFlux;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                   const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                   PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;

    const int T = 0;
    const int VEL = 1;

    PetscErrorCode ierr;

    // set the fluxes to zero
    flux[CompressibleFlowFields::RHOE] = 0.0;

    // Compute the les stress tensor LESTau
    PetscReal lestau[9];  // Maximum size without symmetry
    PetscReal mut;

    ierr = LesViscosity(dim, ctx, fg, field, uOff, mut);
    CHKERRQ(ierr);
    ierr = CompressibleFlowComputeLesStressTensor(dim, fg, gradAux + aOff_x[VEL], uOff_x, field, ctx, lestau);
    CHKERRQ(ierr);

    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal lesHeatFlux = 0.0;
        // add in the contributions for this turbulence terms
        for (PetscInt c = 0; c < dim; ++c) {
            lesHeatFlux += aux[aOff[VEL] + c] * lestau[d * dim + c];
        }

        // LES heat conduction (-kt dT/dx - kt dT/dy - kt  dT/dz) . n A
        lesHeatFlux += c_p * mut * gradAux[aOff_x[T] + d];

        // Multiply by the area normal
        lesHeatFlux *= -fg->normal[d];

        flux[CompressibleFlowFields::RHOE] += lesHeatFlux;
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                    const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                    PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;

    // this order is based upon the order that they are passed into RegisterRHSFunction
    PetscReal mut;
    const int yi = 0;

    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    ierr = LesViscosity(dim, ctx, fg, field, uOff, mut);
    CHKERRQ(ierr);

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            // LESspeciesFlux(-rho mut dYi/dx - mut dYi/dy - rho mut dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal lesSpeciesFlux = -fg->normal[d] * mut * gradAux[offset];
            flux[sp] += lesSpeciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesEvFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                               const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                               PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int euler = 1;
    const int VEL = 1;

    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;


    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // Compute the LES stress tensor tau
    PetscReal lestau[9];  // Maximum size without symmetry
    const int EV_FIELD = 1;
    PetscReal mut;

    ierr = LesViscosity(dim, ctx, fg, field, uOff, mut);
    CHKERRQ(ierr);
    ierr = CompressibleFlowComputeLesStressTensor(dim, fg, gradAux + aOff_x[VEL], uOff, field, ctx, lestau);
    CHKERRQ(ierr);

    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // energy equation
    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        for (PetscInt d = 0; d < dim; ++d) {
            PetscReal lesEvFlux = 0.0;
            for (PetscInt c = 0; c < dim; ++c) {
                //
                lesEvFlux += -areaMag * sqrt(areaMag) * density * lestau[d * dim + c] * lestau[d * dim + c] / mut;
            }

            //  LESevFlux( rho Di dEVi/dx + rho Di dEVi/dy + rho Di dEVi//dz) . n A +  LESevFlux(-rho ce EV^3/2 ) . n A
            const int offset = aOff_x[EV_FIELD]  + d;
            lesEvFlux += -fg->normal[d] * (density * mut * gradAux[offset] - c_e * field[uOff[EV_FIELD]  ] * sqrt(field[uOff[EV_FIELD]  ] / density));

            flux[ev] = lesEvFlux;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::CompressibleFlowComputeLesStressTensor(PetscInt dim, const PetscFVFaceGeom* fg, const PetscReal* gradVel, const PetscInt uOff[],
                                                                                            const PetscScalar field[], void* ctx, PetscReal* lestau) {
    PetscFunctionBeginUser;
    // pre-compute the div of the velocity field
    PetscReal divVel = 0.0;
    PetscReal mut;

    PetscErrorCode ierr;
    ierr = LesViscosity(dim, ctx, fg, field, uOff, mut);
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
                lestau[c * dim + d] = 2.0 * mut * ((gradVel[c * dim + d]) - divVel / 3.0);
            } else {
                // for xy, xz, etc
                lestau[c * dim + d] = mut * ((gradVel[c * dim + d]) + (gradVel[d * dim + c]));
            }
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesViscosity(PetscInt dim, void * ctx, const PetscFVFaceGeom* fg, const PetscScalar field[], const PetscInt uOff[], PetscReal& mut) {
    PetscFunctionBeginUser;
    const int euler = 0;
    const int EV_FIELD = 0;

  //  auto flowParameters = (DiffusionData*)ctx;


    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // get the current ev from ev_field for computing turbulent kinetic energy
    const PetscReal k = field[uOff[EV_FIELD] ];
    // compute LES viscosity
    mut = c_k * sqrt(areaMag * k / density);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::LES, "Creating LES sources for Navier-Stokes Eqs.",
         ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"));
