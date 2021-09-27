#include "stiffenedGas.hpp"

ablate::eos::StiffenedGas::StiffenedGas(std::shared_ptr<ablate::parameters::Parameters> parametersIn, std::vector<std::string> species) : EOS("stiffenedGas"), species(species) {
    // set default values for options
    parameters.gamma = parametersIn->Get<PetscReal>("gamma", 2.4);
    parameters.Cv = parametersIn->Get<PetscReal>("Cv", 3030.0);
    parameters.p0 = parametersIn->Get<PetscReal>("p0", 1.0e7);
    parameters.T0 = parametersIn->Get<PetscReal>("T0", 584.25);
    parameters.e0 = parametersIn->Get<PetscReal>("e0", 1393000.0);
    parameters.numberSpecies = species.size();
}

void ablate::eos::StiffenedGas::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tgamma: " << parameters.gamma << std::endl;
    stream << "\tCv: " << parameters.Cv << std::endl;
    stream << "\tp0: " << parameters.p0 << std::endl;
    stream << "\tT0: " << parameters.T0 << std::endl;
    stream << "\te0: " << parameters.e0 << std::endl;
    if (!species.empty()) {
        stream << "\tspecies: " << species.front();
        for (std::size_t i = 1; i < species.size(); i++) {
            stream << ", " << species[i];
        }
        stream << std::endl;
    }
}

PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasDecodeState(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *velocity, const PetscReal densityYi[],
                                                                  PetscReal *internalEnergy, PetscReal *a, PetscReal *p, void *ctx) {
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
    *p = (parameters->gamma - 1.0) * density * (*internalEnergy) - parameters->gamma * parameters->p0;
    *a = PetscSqrtReal(parameters->gamma * ((*p) + parameters->p0) / density);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeTemperature(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *massFlux, const PetscReal densityYi[], PetscReal *T,
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
    PetscReal cv = parameters->Cv;

    (*T) = (internalEnergy - parameters->e0) / cv + parameters->T0;
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSpeciesSensibleEnthalpy(PetscReal temperature, PetscReal *hi, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    for (PetscInt s = 0; s < parameters->numberSpecies; s++) {
        hi[s] = 0.0;
    }
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeDensityFunctionFromTemperaturePressure(PetscReal temperature, PetscReal pressure, const PetscReal *yi, PetscReal *density, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    *density = (pressure + parameters->gamma * parameters->p0) / ((parameters->gamma - 1) * (parameters->Cv * (temperature - parameters->T0) + parameters->e0));
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSensibleInternalEnergy(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *sensibleInternalEnergy, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    PetscReal cv = parameters->Cv;
    *sensibleInternalEnergy = (T - parameters->T0) * cv + parameters->e0;
    PetscFunctionReturn(0);
}
PetscErrorCode ablate::eos::StiffenedGas::StiffenedGasComputeSpecificHeatConstantPressure(PetscReal T, PetscReal density, const PetscReal *yi, PetscReal *specificHeat, void *ctx) {
    PetscFunctionBeginUser;
    Parameters *parameters = (Parameters *)ctx;
    (*specificHeat) = parameters->gamma * parameters->Cv;
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::StiffenedGas, "stiffened gas eos", ARG(ablate::parameters::Parameters, "parameters", "parameters for the stiffened gas eos"),
         OPT(std::vector<std::string>, "species", "species to track.  Note: species mass fractions do not change eos"));