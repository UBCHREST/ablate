#include "les.hpp"
#include <utility>
#include "finiteVolume/turbulenceFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::finiteVolume::processes::LES::LES(std::string tkeFieldIn) : tkeField(tkeFieldIn.empty() ? ablate::finiteVolume::TurbulenceFlowFields::TKE_FIELD : tkeFieldIn) {}

void ablate::finiteVolume::processes::LES::Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) {
    auto conservedFieldName = CompressibleFlowFields::CONSERVED + tkeField;
    if (flow.GetSubDomain().ContainsField(conservedFieldName)) {
        const auto& tkeFieldDescription = flow.GetSubDomain().GetField(tkeField);
        if (tkeFieldDescription.Tagged(CompressibleFlowFields::EV_TAG) && tkeFieldDescription.numberComponents != 1) {
            throw std::invalid_argument("The ablate::finiteVolume::processes::LES expects the tke field \"" + conservedFieldName + "\" to have one component and be tagged as an EV.");
        }
    } else {
        throw std::invalid_argument("The field must contain the " + conservedFieldName + " field.");
    }

    if (!flow.GetSubDomain().ContainsField(tkeField)) {
        throw std::invalid_argument("The ablate::finiteVolume::processes::LES expects the non-conserved form of tke field \"" + tkeField + "\".");
    }

    // Register the euler/Energy LESdiffusion source term tke
    flow.RegisterRHSFunction(LesEnergyFlux,
                             nullptr,
                             CompressibleFlowFields::EULER_FIELD,
                             {CompressibleFlowFields::EULER_FIELD},
                             {tkeField, CompressibleFlowFields::VELOCITY_FIELD, CompressibleFlowFields::TEMPERATURE_FIELD});

    // Register the N_S LESdiffusion source terms
    flow.RegisterRHSFunction(LesMomentumFlux, nullptr, CompressibleFlowFields::EULER_FIELD, {CompressibleFlowFields::EULER_FIELD}, {tkeField, CompressibleFlowFields::VELOCITY_FIELD});

    // Register the tke LESdiffusion source term
    flow.RegisterRHSFunction(LesTkeFlux, nullptr, conservedFieldName, {CompressibleFlowFields::EULER_FIELD}, {tkeField, CompressibleFlowFields::VELOCITY_FIELD});

    // Register the Species LESdiffusion source term
    if (flow.GetSubDomain().ContainsField(CompressibleFlowFields::DENSITY_YI_FIELD)) {
        // the species are treated like any other transported ev
        auto& numberComponent = numberComponents.emplace_back(flow.GetSubDomain().GetField(CompressibleFlowFields::YI_FIELD).numberComponents);
        flow.RegisterRHSFunction(LesEvFlux, &numberComponent, CompressibleFlowFields::DENSITY_YI_FIELD, {CompressibleFlowFields::EULER_FIELD}, {tkeField, CompressibleFlowFields::YI_FIELD});
    }

    // March over any ev
    for (auto& evConservedField : flow.GetSubDomain().GetFields(domain::FieldLocation::SOL, CompressibleFlowFields::EV_TAG)) {
        // Get the nonConserved form
        auto nonConservedEvName = evConservedField.name.substr(CompressibleFlowFields::CONSERVED.length());

        // the species are treated like any other transported ev
        auto& numberComponent = numberComponents.emplace_back(evConservedField.numberComponents);
        flow.RegisterRHSFunction(LesEvFlux, &numberComponent, evConservedField.name, {CompressibleFlowFields::EULER_FIELD}, {tkeField, nonConservedEvName});
    }
}

PetscErrorCode ablate::finiteVolume::processes::LES::LesMomentumFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt* uOff, const PetscInt* uOff_x, const PetscScalar* field,
                                                                     const PetscScalar* grad, const PetscInt* aOff, const PetscInt* aOff_x, const PetscScalar* aux, const PetscScalar* gradAux,
                                                                     PetscScalar* flux, void* ctx) {
    PetscFunctionBeginUser;
    const int euler = 0;
    const int TKE_FIELD = 0;
    const int VEL = 1;

    for (PetscInt d = 0; d < dim; d++) {
        flux[CompressibleFlowFields::RHOU + d] = 0.0;
    }
    PetscReal tau[9];
    PetscReal muT;
    PetscReal turbulence;

    turbulence = aux[aOff[TKE_FIELD]];

    PetscCall(LesViscosity(dim, fg, field + uOff[euler], turbulence, muT));
    // Compute the LES stress tensor tau
    PetscCall(NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, muT, gradAux + aOff_x[VEL], tau));

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
    const int TKE_FIELD = 0;
    const int VEL = 1;
    const int T = 2;

    // set the fluxes to zero
    flux[CompressibleFlowFields::RHOE] = 0.0;

    PetscReal tau[9];
    PetscReal muT;

    PetscReal turbulence = aux[aOff[TKE_FIELD]];

    // Compute the les stress tensor LESTau
    PetscCall(LesViscosity(dim, fg, field + uOff[euler], turbulence, muT));
    PetscCall(NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, muT, gradAux + aOff_x[VEL], tau));

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
PetscErrorCode ablate::finiteVolume::processes::LES::LesTkeFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;
    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int euler = 0;
    const int TKE_FIELD = 0;
    const int VEL = 1;

    // get the current density from euler
    const PetscReal density = field[uOff[euler] + CompressibleFlowFields::RHO];

    PetscReal tau[9];
    PetscReal muT;

    PetscReal turbulence = aux[aOff[TKE_FIELD]];
    PetscCall(LesViscosity(dim, fg, field + uOff[euler], turbulence, muT));

    // Compute the LES stress tensor tau
    PetscCall(NavierStokesTransport::CompressibleFlowComputeStressTensor(dim, 1, gradAux + aOff_x[VEL], tau));
    const PetscReal areaMag = utilities::MathUtilities::MagVector(dim, fg->normal);

    flux[0] = 0;
    for (PetscInt d = 0; d < dim; ++d) {
        PetscReal lesEvFlux_0 = 0;
        PetscReal lesEvFlux_1;
        PetscReal lesEvFlux;

        //  calculate turbulent diff other tke
        for (PetscInt c = 0; c < dim; ++c) {
            lesEvFlux_0 += -0.5 * fg->normal[d] * sqrt(areaMag) * density * muT * tau[d * dim + c] * tau[d * dim + c];
        }
        const int offset = aOff_x[TKE_FIELD] + d;

        lesEvFlux_1 = density * muT * gradAux[offset] - c_e * density * turbulence * sqrt(abs(turbulence));

        lesEvFlux_1 *= -fg->normal[d];
        lesEvFlux = lesEvFlux_0 + lesEvFlux_1;
        flux[0] += lesEvFlux;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::finiteVolume::processes::LES::LesEvFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                               const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                               PetscScalar flux[], void* ctx) {
    PetscFunctionBeginUser;

    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int euler = 0;
    const int TKE_FIELD = 0;
    const int EV_FIELD = 1;

    PetscReal muT;
    auto numberComponents = *(PetscInt*)ctx;
    PetscReal turbulence = aux[aOff[TKE_FIELD]];
    PetscCall(LesViscosity(dim, fg, field + uOff[euler], turbulence, muT));

    // species/ev equations
    for (PetscInt sp = 0; sp < numberComponents; ++sp) {
        PetscReal lesSpeciesFlux;
        flux[sp] = 0;

        for (PetscInt d = 0; d < dim; ++d) {
            // LESspeciesFlux(-rho mut dYi/dx - mut dYi/dy - rho mut dYi//dz) . n A
            const int offset = aOff_x[EV_FIELD] + (sp * dim) + d;
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
         OPT(std::string, "tke", "optional name of turbulent kinetic energy field (default is TurbulenceFlowFields::TKE_FIELD)"));