#include "les.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::processes::LES::LES(std::string tke, std::shared_ptr<eos::EOS> eosIn) : tke(std::move(tke)), eos(std::move(eosIn)) {}

void ablate::finiteVolume::processes::LES::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_EV_FIELD)) {
        if (!flow.GetSubDomain().ContainsField(CompressibleFlowFields::EV_FIELD)) {                 //Do I need to throw an error here?
            throw std::invalid_argument("The ablate::finiteVolume::processes::EVTransport process expects the conserved (" + CompressibleFlowFields::DENSITY_EV_FIELD + ") and non-conserved (" +
                                        CompressibleFlowFields::EV_FIELD + ") extra variables to be in the flow.");
        }
        //
        auto conservedForm = flow.GetSubDomain().GetField(CompressibleFlowFields::DENSITY_EV_FIELD);

        const auto& extraVariableList = conservedForm.components;

        diffusionData.tke_ev = -1;
        for (std::size_t ev = 0; ev < extraVariableList.size(); ev++) {
            if (extraVariableList[ev] == tke) {
                diffusionData.tke_ev = ev;
            }
        }
        if (diffusionData.tke_ev < 0) {
            throw std::invalid_argument("The LES solver cannot find the " + tke + "");           // takes any ev as tke
        }

        flow.RegisterRHSFunction(
            LesMomentumFlux, &diffusionData, CompressibleFlowFields::EULER_FIELD, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::EV_FIELD, CompressibleFlowFields::VELOCITY_FIELD});
        // Register the euler/Energy LESdiffusion source termtke
        flow.RegisterRHSFunction(LesEnergyFlux,
                                 &diffusionData,
                                 CompressibleFlowFields::EULER_FIELD,
                                 {CompressibleFlowFields::EULER_FIELD},
                                 {CompressibleFlowFields::EV_FIELD, CompressibleFlowFields::VELOCITY_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});

        // Register the ev LESdiffusion source term
        flow.RegisterRHSFunction(
            LesEvFlux, &diffusionData, CompressibleFlowFields::DENSITY_EV_FIELD, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::EV_FIELD, CompressibleFlowFields::VELOCITY_FIELD});

        // Register the Species LESdiffusion source term
        if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::YI_FIELD)) {
            diffusionData.numberSpecies = (PetscInt)eos->GetSpecies().size();         // Am I getting Species number?

            flow.RegisterRHSFunction(
                LesSpeciesFlux, &diffusionData, CompressibleFlowFields::DENSITY_YI_FIELD, {CompressibleFlowFields::EULER_FIELD}, {CompressibleFlowFields::EV_FIELD, CompressibleFlowFields::YI_FIELD});
        }
    }
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesMomentumFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x, const PetscScalar* field,
                                                                     const PetscScalar* grad, const PetscInt* aOff, const PetscInt* aOff_x, const PetscScalar* aux, const PetscScalar* gradAux,
                                                                     PetscScalar* flux, void* ctx) {
    PetscFunctionBeginUser;
    const int euler = 0;
    const int EV_FIELD = 0;
    const int VEL = 1;

    for (PetscInt d = 0; d < dim; d++) {
        flux[CompressibleFlowFields::RHOU + d] = 0.0;
    }
    // Compute the LES stress tensor tau
    PetscReal lestau[9];  // Maximum size without symmetry

    PetscErrorCode ierr;
    ierr = CompressibleFlowComputeLesStressTensor(dim, ctx, fg, field + uOff[euler], aux + aOff[EV_FIELD], gradAux + aOff_x[VEL], lestau); // ctx, fg?
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
    const int euler = 0;
    const int EV_FIELD = 0;
    const int VEL = 1;
    const int T = 2;

    PetscErrorCode ierr;

    // set the fluxes to zero
    flux[CompressibleFlowFields::RHOE] = 0.0;

    // Compute the les stress tensor LESTau
    PetscReal lestau[9];  // Maximum size without symmetry
    PetscReal mut;

    ierr = LesViscosity(dim, ctx, fg, field + uOff[euler], aux + aOff[EV_FIELD], mut);
    CHKERRQ(ierr);
    ierr = CompressibleFlowComputeLesStressTensor(dim, ctx, fg, field + uOff[euler], aux + aOff[EV_FIELD], gradAux + aOff_x[VEL], lestau);
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
PetscErrorCode ablate::finiteVolume::processes::LES::LesEvFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                               const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                               PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int euler = 0;
    const int EV_FIELD = 0;
        const int VEL = 1;

    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // Compute the LES stress tensor tau
    PetscReal lestau[9];  // Maximum size without symmetry
    PetscReal mut;

    ierr = LesViscosity(dim, ctx, fg, field + uOff[euler], aux + aOff[EV_FIELD], mut);
    CHKERRQ(ierr);
    ierr = CompressibleFlowComputeLesStressTensor(dim, ctx, fg, field + uOff[euler], aux + aOff[EV_FIELD], gradAux + aOff_x[VEL], lestau);
    CHKERRQ(ierr);
    
        for (PetscInt ev = 0; ev < flowParameters->numberSpecies; ++ev) {
        flux[sp] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            // LESspeciesFlux(-rho mut dYi/dx - mut dYi/dy - rho mut dYi//dz) . n A
            const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;
            PetscReal lesSpeciesFlux = -fg->normal[d] * mut * gradAux[offset];
            flux[sp] += lesSpeciesFlux;
        }
    }
    flux[flowParameters->tke_ev] = 0;
        const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);
    // energy equation
    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal lesEvFlux = 0.0;
        for (PetscInt c = 0; c < dim; ++c) {
            //
            lesEvFlux += sqrt(areaMag) * density * lestau[d * dim + c] * lestau[d * dim + c] / mut ;
        }

        //  LESevFlux( rho Di dEVi/dx + rho Di dEVi/dy + rho Di dEVi//dz) . n A +  LESevFlux(-rho ce EV^3/2 ) . n A

        const int offset = aOff_x[EV_FIELD] + (flowParameters->tke_ev * dim) + d;        // only counting tke here
        lesEvFlux += density * mut * gradAux[offset] - c_e * density * aux[aOff[EV_FIELD] + flowParameters->tke_ev] * sqrt(aux[aOff[EV_FIELD] + flowParameters->tke_ev]);
        lesEvFlux *= -fg->normal[d];

        flux[flowParameters->tke_ev] += lesEvFlux;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::LES::LesSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                    const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                    PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;

    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int euler = 0;
    const int EV_FIELD = 0;
    const int yi = 1;

    PetscReal mut;
    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    ierr = LesViscosity(dim, ctx, fg, field + uOff[euler], aux + aOff[EV_FIELD], mut);
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

PetscErrorCode ablate::finiteVolume::processes::LES::CompressibleFlowComputeLesStressTensor(PetscInt dim, void* ctx, const PetscFVFaceGeom* fg, const PetscScalar* densityField,
                                                                                            const PetscScalar* getTke, const PetscReal* gradVel, PetscReal* lestau) {
    PetscFunctionBeginUser;

    // pre-compute the div of the velocity field
    PetscReal divVel = 0.0;
    PetscReal mut;

    PetscErrorCode ierr;
    ierr = LesViscosity(dim, ctx, fg, densityField, getTke, mut);
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

PetscErrorCode ablate::finiteVolume::processes::LES::LesViscosity(PetscInt dim, void* ctx, const PetscFVFaceGeom* fg, const PetscScalar* densityField, const PetscScalar* getTke, PetscReal& mut) {
    PetscFunctionBeginUser;

    auto flowParameters = (DiffusionData*)ctx;    // give up passing fields into functions

    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // get the current density from euler
    const PetscReal density = densityField[CompressibleFlowFields::RHO];

    // get the current ev from ev_field for computing turbulent kinetic energy
    const PetscReal turbulence = getTke[flowParameters->tke_ev];
    // compute LES viscosity
    mut = c_k * density * sqrt(areaMag * turbulence);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::LES, "Creating LES sources for Navier-Stokes Eqs.",
         ARG(std::string, "tke", "the name of turbulent kinetic energy "), ARG(ablate::eos::EOS, "eos", "the equation of state used to describe the flow"));

