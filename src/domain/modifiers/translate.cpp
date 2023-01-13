#include "translate.hpp"
#include <algorithm>
#include "mathFunctions/functionFactory.hpp"

ablate::domain::modifiers::Translate::Translate(std::vector<double> translateIn) : ablate::domain::modifiers::MeshMapper(mathFunctions::Create(TranslateFunction, translate)) {
    std::copy_n(translateIn.begin(), std::min(translateIn.size(), (std::size_t)3), std::begin(translate));
}
std::string ablate::domain::modifiers::Translate::ToString() const {
    return "ablate::domain::modifiers::Translate " + std::to_string(translate[0]) + ", " + std::to_string(translate[1]) + ", " + std::to_string(translate[2]);
}

PetscErrorCode ablate::domain::modifiers::Translate::TranslateFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto translate = (PetscReal *)ctx;
    for (PetscInt d = 0; d < Nf; d++) {
        u[d] = x[d] + translate[d];
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::Translate, "Translate the x,y,z coordinate of the domain mesh by the values.", std::vector<double>);
