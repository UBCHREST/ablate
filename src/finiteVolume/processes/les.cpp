#include "les.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"
#include "navierStokesTransport.hpp"


ablate::finiteVolume::processes::LES::LES(std::string tke) : tke(std::move(tke)) {}

void ablate::finiteVolume::processes::LES::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_EV_FIELD)) {
        if (!flow.GetSubDomain().ContainsField(CompressibleFlowFields::EV_FIELD)) {                 //Do I need to throw an error here?
            throw std::invalid_argument("The ablate::finiteVolume::processes::EVTransport process expects the conserved (" + CompressibleFlowFields::DENSITY_EV_FIELD + ") and non-conserved (" +
                                        CompressibleFlowFields::EV_FIELD + ") extra variables to be in the flow.");
        }
        //
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
    // Compute the LES stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscReal mut;
    PetscReal     turbulence;
    auto flowParameters = (DiffusionData*)ctx;

    turbulence = aux[ aOff[EV_FIELD]+ flowParameters->tke_ev];
    PetscErrorCode ierr;

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
    CHKERRQ(ierr);
    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, mut, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);

    // for each velocity component
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

    // Compute the les stress tensor LESTau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscReal mut;
    PetscReal     turbulence;
    auto flowParameters = (DiffusionData*)ctx;

    turbulence = aux[ aOff[EV_FIELD]+ flowParameters->tke_ev];

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
    CHKERRQ(ierr);
    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, mut, gradAux + aOff_x[VEL], tau);
    CHKERRQ(ierr);

    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal lesHeatFlux = 0.0;
        // add in the contributions for this turbulence terms
        for (PetscInt c = 0; c < dim; ++c) {
            lesHeatFlux += aux[aOff[VEL] + c] * tau[d * dim + c];
        }

        // LES heat conduction (-kt dT/dx - kt dT/dy - kt  dT/dz) . n A
        lesHeatFlux += c_p * mut * gradAux[aOff_x[T] + d] / prT;

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
    PetscReal turbulence;
    auto flowParameters = (DiffusionData*)ctx;
    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    // Compute the LES stress tensor tau
    PetscReal tau[9];  // Maximum size without symmetry
    PetscReal mut;

    turbulence = aux[ aOff[EV_FIELD]+ flowParameters->tke_ev];

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
    CHKERRQ(ierr);

    ierr = flowParameters->computeTau->CompressibleFlowComputeStressTensor(dim, mut, gradAux + aOff_x[VEL], tau);
        CHKERRQ(ierr);
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);
    // energy equation

    for (PetscInt ev = 0; ev < flowParameters->numberEV; ++ev) {
        flux[flowParameters->tke_ev] = 0;
        flux[ev] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            PetscReal lesEvFlux = 0.0;
            for (PetscInt c = 0; c < dim; ++c) {
                //

                lesEvFlux += sqrt(areaMag) * density * tau[d * dim + c] * tau[d * dim + c] / mut;
            }

            //  LESevFlux( rho Di dEVi/dx + rho Di dEVi/dy + rho Di dEVi//dz) . n A +  LESevFlux(-rho ce EV^3/2 ) . n A

            const int offset = aOff_x[EV_FIELD] + (ev * dim) + d;
            PetscReal evFlux = -fg->normal[d] * mut * gradAux[offset] / scT;
            flux[ev] += evFlux;

            const int offset_tke = aOff_x[EV_FIELD] + (flowParameters->tke_ev * dim) + d;  // only counting tke here
            lesEvFlux += density * mut * gradAux[offset_tke] - c_e * density * aux[aOff[EV_FIELD] + flowParameters->tke_ev] * sqrt(aux[aOff[EV_FIELD] + flowParameters->tke_ev]);
            lesEvFlux *= -fg->normal[d];

            flux[flowParameters->tke_ev] += lesEvFlux;
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

     PetscReal turbulence;
     PetscReal mut;
    PetscErrorCode ierr;
    auto flowParameters = (DiffusionData*)ctx;

    turbulence = aux[ aOff[EV_FIELD]+ flowParameters->tke_ev];

    ierr = LesViscosity(dim, fg, field + uOff[euler], turbulence, mut);
    CHKERRQ(ierr);

    // species equations
    for (PetscInt sp = 0; sp < flowParameters->numberSpecies; ++sp) {
        flux[sp] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            // LESspeciesFlux(-rho mut dYi/dx - mut dYi/dy - rho mut dYi//dz) . n A
            const int offset = aOff_x[yi] + (sp * dim) + d;
            PetscReal lesSpeciesFlux = -fg->normal[d] * mut * gradAux[offset] / scT;
            flux[sp] += lesSpeciesFlux;
        }
    }

    PetscFunctionReturn(0);
}


PetscErrorCode ablate::finiteVolume::processes::LES::LesViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar* densityField, const  PetscReal turbulence, PetscReal& mut) {
    PetscFunctionBeginUser;

    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    // get the current density from euler
    const PetscReal density = densityField[CompressibleFlowFields::RHO];

    mut = c_k * density * sqrt(areaMag * turbulence);

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::LES, "Creating LES sources for Navier-Stokes Eqs.",
         ARG(std::string, "tke", "the name of turbulent kinetic energy "));

