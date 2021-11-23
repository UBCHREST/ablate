#include "perfectGas.hpp"

ablate::eos::PerfectGas::PerfectGas(std::shared_ptr<ablate::parameters::Parameters> parametersIn, std::vector<std::string> species) : EOS("perfectGas"), species(species) {
    // set default values for options
    parameters.gamma = parametersIn->Get<PetscReal>("gamma", 1.4);
    parameters.rGas = parametersIn->Get<PetscReal>("Rgas", 287.0);
    parameters.numberSpecies = species.size();
}

void ablate::eos::PerfectGas::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tgamma: " << parameters.gamma << std::endl;

    stream << "\tRgas: " << parameters.rGas << std::endl;
    if (!species.empty()) {
        stream << "\tspecies: " << species.front();
        for (std::size_t i = 1; i < species.size(); i++) {
            stream << ", " << species[i];
        }
        stream << std::endl;
    }
}

PetscErrorCode ablate::eos::PerfectGas::PerfectGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *velocity, const PetscReal densityYi[], PetscReal *internalEnergy,
                                                              PetscReal *a, PetscReal *p, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;

    // Get the velocity in this direction
    PetscReal ke = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        ke += PetscSqr(velocity[d]);
    }
    ke *= 0.5;

    // assumed eos
    (*internalEnergy) = (totalEnergy)-ke;
    *p = (parameters->gamma - 1.0) * density * (*internalEnergy);
    *a = PetscSqrtReal(parameters->gamma * (*p) / density);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *massFlux, const PetscReal densityYi[], PetscReal *T,
                                                                     void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;

    // Get the velocity in this direction
    PetscReal speedSquare = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        speedSquare += PetscSqr(massFlux[d] / density);
    }

    // assumed eos
    PetscReal internalEnergy = (totalEnergy)-0.5 * speedSquare;
    PetscReal cv = parameters->rGas / (parameters->gamma - 1.0);

    (*T) = internalEnergy / cv;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSpeciesSensibleEnthalpy(PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    for (PetscInt s = 0; s < parameters->numberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeDensityFunctionFromTemperaturePressure(PetscReal temperature, PetscReal pressure, const PetscReal *yi, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    *density = pressure / (temperature * parameters->rGas);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    PetscReal cv = parameters->rGas / (parameters->gamma - 1.0);
    *sensibleInternalEnergy = T * cv;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSpecificHeatConstantPressure(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    (*specificHeat) = parameters->gamma * parameters->rGas / (parameters->gamma - 1.0);
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeSpecificHeatConstantVolume(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    (*specificHeat) = parameters->rGas / (parameters->gamma - 1.0);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::PerfectGas, "perfect gas eos", ARG(ablate::parameters::Parameters, "parameters", "parameters for the perfect gas eos"),
         OPT(std::vector<std::string>, "species", "species to track.  Note: species mass fractions do not change eos"));