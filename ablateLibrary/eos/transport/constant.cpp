#include "constant.hpp"

ablate::eos::transport::Constant::Constant(double k, double mu, double diff): active(k || mu || diff), k(k), mu(mu), diff(diff){}

void ablate::eos::transport::Constant::ConstantFunction(PetscReal, PetscReal, const PetscReal*, PetscReal &result, void *ctx) {
    result = *(double*)ctx;
}

#include "parser/registrar.hpp"
REGISTERDEFAULT(ablate::eos::transport::TransportModel, ablate::eos::transport::Constant, "constant value transport model (often used for testing)",
                OPT(double, "k", "thermal conductivity [W/(m K)]"),
                OPT(double, "mu", "viscosity [Pa s]"),
                OPT(double, "diff", "diffusivity [m2/s]"));