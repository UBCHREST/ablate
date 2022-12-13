#include "les.hpp"
#include <utility>
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::processes::LES::LES(std::string tke) : tke(std::move(tke)) {}

void ablate::finiteVolume::processes::LES::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_EV_FIELD)) {
        if (!flow.GetSubDomain().ContainsField(CompressibleFlowFields::EV_FIELD)) {
            throw std::invalid_argument("The ablate::finiteVolume::processes::EVTransport process expects the conserved (" + CompressibleFlowFields::DENSITY_EV_FIELD + ") and non-conserved (" +
                                        CompressibleFlowFields::EV_FIELD + ") extra variables to be in the flow.");
        }
        auto conservedForm = flow.GetSubDomain().GetField(CompressibleFlowFields::DENSITY_EV_FIELD);
        diffusionData.numberEV = conservedForm.numberComponents;

        const auto& extraVariableList = conservedForm.components;

        diffusionData.tke_ev = -1;

        for (std::size_t ev = 0; ev < extraVariableList.size(); ev++) {
            if (extraVariableList[ev] == tke) {
                diffusionData.tke_ev = (PetscInt)ev;
            }
        }
        if (diffusionData.tke_ev < 0) {
            throw std::invalid_argument("The LES solver cannot find the tke");
        }
        if (tke.empty()) {
            throw std::invalid_argument("The LES solver needs an extraVariable as " + tke + "");
        }
        // Register the N_S LESdiffusion source terms
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
        if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_YI_FIELD)) {
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
    PetscReal tau[9];
    PetscReal muT;
    PetscReal turbulence;
    auto flowParameters = (DiffusionData*)ctx;

    turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];
    PetscErrorCode ierr;

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
    CHKERRQ(ierr);
    // Compute the LES stress tensor tau
    ierr = NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, muT, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);

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

    PetscReal tau[9];
    PetscReal muT;
    auto flowParameters = (DiffusionData*)ctx;

    PetscReal turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];

    // Compute the les stress tensor LESTau
    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
    CHKERRQ(ierr);
    ierr = NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, muT, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);

    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal lesHeatFlux = 0.0;
        // add in the contributions for this turbulence terms
        for (PetscInt c = 0; c < dim; ++c) {
            lesHeatFlux += aux[aOff[VEL] + c] * tau[d * dim + c];
        }

        // LES heat conduction (-kt dT/dx - kt dT/dy - kt  dT/dz) . n A
        lesHeatFlux += c_p * muT * gradAux[aOff_x[T] + d] / prT;

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

    PetscReal tau[9];
    PetscReal muT;

    PetscReal turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
    CHKERRQ(ierr);

    // Compute the LES stress tensor tau
    ierr = NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, 1, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        flux[ev] = 0;
        for (PetscInt d = 0; d < dim; ++d) {
            PetscReal lesEvFlux_0 = 0;
            PetscReal lesEvFlux_1;
            PetscReal lesEvFlux;

            //  add turbulent diff other ev
            if (ev != flowParameters->tke_ev) {
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

    PetscReal muT;
    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    PetscReal turbulence = aux[aOff[EV_FIELD] + flowParameters->tke_ev];

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, muT);
    CHKERRQ(ierr);

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        PetscReal lesSpeciesFlux;
        flux[sp] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            // LESspeciesFlux(-rho mut dYi/dx - mut dYi/dy - rho mut dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            lesSpeciesFlux = -fg->normal[d] * muT * gradAux[offset] / scT;
            flux[sp] += lesSpeciesFlux;
        }
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar* densityField, const PetscReal turbulence, PetscReal& muT) {
    PetscFunctionBeginUser;

    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // get the current density from euler
    const PetscReal density = densityField[CompressibleFlowFields::RHO];

    // Compute the LES turbulent viscosity
    muT = c_k * density * sqrt(abs(areaMag * turbulence));

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::LES, "Creating LES sources for Navier-Stokes Eqs.",
         ARG(std::string, "tke", "the name of turbulent kinetic energy "));