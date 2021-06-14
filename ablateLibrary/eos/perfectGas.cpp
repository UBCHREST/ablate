#include "perfectGas.hpp"

ablate::eos::PerfectGas::PerfectGas(std::shared_ptr<ablate::parameters::Parameters> parametersIn) : EOS("perfectGas") {
    // set default values for options
    parameters.gamma = parametersIn->Get<PetscReal>("gamma", 1.4);
    parameters.rGas = parametersIn->Get<PetscReal>("Rgas", 287.0);
}

void ablate::eos::PerfectGas::View(std::ostream &stream) const {
    stream << "EOS: " << type << std::endl;
    stream << "\tgamma: " << parameters.gamma << std::endl;
    stream << "\tRgas: " << parameters.rGas << std::endl;
}

PetscErrorCode ablate::eos::PerfectGas::PerfectGasDecodeState(const PetscReal *yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *velocity, PetscReal *internalEnergy,
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

PetscErrorCode ablate::eos::PerfectGas::PerfectGasComputeTemperature(const PetscReal *yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal *massFlux, PetscReal *T, void *ctx) {
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

#include "parser/registrar.hpp"
REGISTER(ablate::eos::EOS, ablate::eos::PerfectGas, "perfect gas eos", ARG(ablate::parameters::Parameters, "parameters", "parameters for the perfect gas eos"));