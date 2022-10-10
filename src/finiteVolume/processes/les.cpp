#include "les.hpp"
<<<<<<< HEAD
=======
#include <utility>
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

<<<<<<< HEAD
ablate::finiteVolume::processes::LES::LES(std::string tke) : tke(std::move(tke)) {}
=======
ablate::finiteVolume::processes::LES::LES(std::string tke) : tke(tke) {}
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553

void ablate::finiteVolume::processes::LES::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_EV_FIELD)) {
        if (!flow.GetSubDomain().ContainsField(CompressibleFlowFields::EV_FIELD)) {
            throw std::invalid_argument("The ablate::finiteVolume::processes::EVTransport process expects the conserved (" + CompressibleFlowFields::DENSITY_EV_FIELD + ") and non-conserved (" +
                                        CompressibleFlowFields::EV_FIELD + ") extra variables to be in the flow.");
        }
<<<<<<< HEAD
        //
=======
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
        auto conservedForm = flow.GetSubDomain().GetField(CompressibleFlowFields::DENSITY_EV_FIELD);
        diffusionData.numberEV = conservedForm.numberComponents;

        const auto& extraVariableList = conservedForm.components;

        diffusionData.tke_ev = -1;

        for (std::size_t ev = 0; ev < extraVariableList.size(); ev++) {
            if (extraVariableList[ev] == tke) {
                diffusionData.tke_ev = ev;
            }
        }
        if (diffusionData.tke_ev < 0) {
            throw std::invalid_argument("The LES solver cannot find the tke");
        }
        if (tke.empty()) {
            throw std::invalid_argument("The LES solver needs an extraVariable as " + tke + "");
        }
<<<<<<< HEAD

=======
        // Register the N_S LESdiffusion source terms
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
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
<<<<<<< HEAD
        if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::YI_FIELD)) {
=======
        if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_YI_FIELD)) {
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
            auto yiCount = flow.GetSubDomain().GetField(CompressibleFlowFields::YI_FIELD);
            diffusionData.numberSpecies = yiCount.numberComponents;
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
<<<<<<< HEAD
    // Compute the LES stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscReal mut;
=======
    PetscReal tau[9];
    PetscReal muT;
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
    PetscReal turbulence;
    auto flowParameters = (DiffusionData*)ctx;

    turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];
    PetscErrorCode ierr;

<<<<<<< HEAD
    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
    CHKERRQ(ierr);
    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, mut, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);

    // for each velocity component
=======
    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
    CHKERRQ(ierr);
    // Compute the LES stress tensor tau
    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, muT, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);

>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
    for (PetscInt c = 0; c < dim; ++c) {
        PetscReal lesViscousFlux = 0.0;

        // March over each direction
        for (PetscInt d = 0; d < dim; ++d) {
            lesViscousFlux += -fg->normal[d] * 2 * tau[c * dim + d];  // This is lesTau[c][d]
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

<<<<<<< HEAD
    // Compute the les stress tensor LESTau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscReal mut;
=======
    PetscReal tau[9];
    PetscReal muT;
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
    auto flowParameters = (DiffusionData*)ctx;

    PetscReal turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];

<<<<<<< HEAD
    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
    CHKERRQ(ierr);
    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, mut, gradAux + aOff_x[VEL], tau);
=======
    // Compute the les stress tensor LESTau
    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
    CHKERRQ(ierr);
    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, muT, gradAux + aOff_x[VEL], tau);
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
    CHKERRQ(ierr);

    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal lesHeatFlux = 0.0;
        // add in the contributions for this turbulence terms
        for (PetscInt c = 0; c < dim; ++c) {
            lesHeatFlux += aux[aOff[VEL] + c] * tau[d * dim + c];
        }

        // LES heat conduction (-kt dT/dx - kt dT/dy - kt  dT/dz) . n A
<<<<<<< HEAD
        lesHeatFlux += c_p * mut * gradAux[aOff_x[T] + d] / prT;

        // Multiply by the area normal
        lesHeatFlux *= -fg->normal[d];

=======
        lesHeatFlux += c_p * muT * gradAux[aOff_x[T] + d] / prT;

        // Multiply by the area normal
        lesHeatFlux *= -fg->normal[d];
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
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

<<<<<<< HEAD
    // Compute the LES stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscReal mut;

    PetscReal turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
    CHKERRQ(ierr);

    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, mut, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);
    // energy equation

    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        flux[ev] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            PetscReal lesEvFlux = 0.0;
            PetscReal evFlux;
            for (PetscInt c = 0; c < dim; ++c) {
                lesEvFlux += 0.5 * sqrt(areaMag) * density * tau[d * dim + c] * tau[d * dim + c] / (mut + 0.00001);
            }

            //  LESevFlux( rho Di dEVi/dx +git check rho Di dEVi/dy + rho Di dEVi//dz) . n A +  LESevFlux(-rho ce EV^3/2 ) . n A
            if (!(ev == flowParameters->tke_ev)) {
                const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;
                evFlux = -fg->normal[d] * mut * gradAux[offset] / scT;
                flux[ev] += evFlux;
            }

            if (ev == flowParameters->tke_ev) {
                const int offset_t = aOff_x[EV_FIELD] + (ev * dim) + d;
                const int offset_tk = aOff[EV_FIELD] + (ev * dim) + d;

                // only counting tke here
                lesEvFlux += density * mut * gradAux[offset_t] - c_e * density * aux[offset_tk] * sqrt(abs(aux[offset_tk]));
                lesEvFlux *= -fg->normal[d];

                flux[ev] += lesEvFlux;
            }
=======
    PetscReal tau[9];
    PetscReal muT;

    PetscReal turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
    CHKERRQ(ierr);

    // Compute the LES stress tensor tau
    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, 1, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        flux[ev] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            PetscReal lesEvFlux_0 = 0;
            PetscReal lesEvFlux_1;
            PetscReal lesEvFlux;

            //  add turbulent diff other ev
            if (!(ev == flowParameters->tke_ev)) {
                const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;
                lesEvFlux = -fg->normal[d] * muT * gradAux[offset] / scT;
            }
            //  calculate turbulent diff other tke
            if (ev == flowParameters->tke_ev) {
                for (PetscInt c = 0; c < dim; ++c) {
                    lesEvFlux_0 += -0.5 * fg->normal[d] * sqrt(areaMag) * density * muT * tau[d * dim + c] * tau[d * dim + c];
                }
                const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;

                lesEvFlux_1 = density * muT * gradAux[offset] - c_e * density * turbulence * sqrt(abs(turbulence));

                lesEvFlux_1 *= -fg->normal[d];
                lesEvFlux = lesEvFlux_0 + lesEvFlux_1;
            }
            flux[ev] += lesEvFlux;
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
        }
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

<<<<<<< HEAD
    PetscReal mut;
=======
    PetscReal muT;
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    PetscReal turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];

<<<<<<< HEAD
    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
=======
    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
    CHKERRQ(ierr);

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
<<<<<<< HEAD
=======
        PetscReal lesSpeciesFlux;
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
        flux[sp] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            // LESspeciesFlux(-rho mut dYi/dx - mut dYi/dy - rho mut dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
<<<<<<< HEAD
            PetscReal lesSpeciesFlux = -fg->normal[d] * mut * gradAux[offset] / scT;
=======
            lesSpeciesFlux = -fg->normal[d] * muT * gradAux[offset] / scT;
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
            flux[sp] += lesSpeciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

<<<<<<< HEAD
PetscErrorCode ablate::finiteVolume::processes::LES::LesViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar* densityField, const PetscReal turbulence, PetscReal& mut) {
=======
PetscErrorCode ablate::finiteVolume::processes::LES::LesViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar* densityField, const PetscReal turbulence, PetscReal& muT) {
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553
    PetscFunctionBeginUser;

    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // get the current density from euler
    const PetscReal density = densityField[CompressibleFlowFields::RHO];

<<<<<<< HEAD
    mut = c_k * density * sqrt(abs(areaMag * turbulence));
=======
    // Compute the LES turbulent viscosity
    muT = c_k * density * sqrt(abs(areaMag * turbulence));
>>>>>>> 244c9f43635e4123c77bd90ee546318d54d42553

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::LES, "Creating LES sources for Navier-Stokes Eqs.",
         ARG(std::string, "tke", "the name of turbulent kinetic energy "));
