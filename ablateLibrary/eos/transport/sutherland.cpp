#include "sutherland.hpp"
ablate::eos::transport::Sutherland::Sutherland(std::shared_ptr<eos::EOS> eosIn)
    : eos(eosIn), cpFunction(eos->GetComputeSpecificHeatConstantPressureFunction()), cpContext(eos->GetComputeSpecificHeatConstantPressureContext()) {}

void ablate::eos::transport::Sutherland::SutherlandComputeConductivityFunction(PetscReal temperature, PetscReal density, const PetscReal *yi, PetscReal &conductivity, void *ctx) {
    // compute the cp as a function of
    auto sutherland = (ablate::eos::transport::Sutherland *)ctx;
    PetscReal cp;
    sutherland->cpFunction(temperature, density, yi, &cp, sutherland->cpContext);

    // compute mu
    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    conductivity = mu * cp / pr;
}
void ablate::eos::transport::Sutherland::SutherlandComputeViscosityFunction(PetscReal temperature, PetscReal, const PetscReal *, PetscReal &viscosity, void *ctx) {
    viscosity = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
}
void ablate::eos::transport::Sutherland::SutherlandComputeDiffusivityFunction(PetscReal temperature, PetscReal density, const PetscReal *yi, PetscReal &diffusivity, void *ctx) {
    double mu = muo * PetscSqrtReal(temperature / to) * (temperature / to) * (to + so) / (temperature + so);
    diffusivity = mu / density / sc;
}

#include "parser/registrar.hpp"
REGISTER(ablate::eos::transport::TransportModel, ablate::eos::transport::Sutherland, "Sutherland Transport model",
         ARG(ablate::eos::EOS, "eos", "The EOS used to compute Cp (needed for Conductivity)"));