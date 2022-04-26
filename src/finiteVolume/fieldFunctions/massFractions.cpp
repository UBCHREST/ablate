#include "massFractions.hpp"
#include <algorithm>
#include <mathFunctions/functionPointer.hpp>

ablate::finiteVolume::fieldFunctions::MassFractions::MassFractions(std::shared_ptr<ablate::eos::EOS> eos, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> massFractionFieldFunctionsIn)
    : ablate::mathFunctions::FieldFunction("yi", std::make_shared<ablate::mathFunctions::FunctionPointer>(ablate::finiteVolume::fieldFunctions::MassFractions::ComputeYiFunction, this)),
      massFractionFieldFunctions(massFractionFieldFunctionsIn) {
    const auto &species = eos->GetSpecies();

    // Map the mass fractions to species
    massFractionFunctions.resize(species.size(), nullptr);

    // march over every yiFunctionIn
    for (const auto &yiFunction : massFractionFieldFunctions) {
        auto it = std::find(species.begin(), species.end(), yiFunction->GetName());

        if (it != species.end()) {
            massFractionFunctions[std::distance(species.begin(), it)] = yiFunction->GetFieldFunction();
        } else {
            throw std::invalid_argument("Cannot find field species " + yiFunction->GetName());
        }
    }
}
PetscErrorCode ablate::finiteVolume::fieldFunctions::MassFractions::ComputeYiFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *yi, void *ctx) {
    PetscFunctionBeginUser;
    auto massFractions = (ablate::finiteVolume::fieldFunctions::MassFractions *)ctx;
    // compute the mass fraction at this location
    try {
        // Take the norm of the species
        PetscScalar yiSum = 0.0;

        for (PetscInt s = 0; s < PetscMin(Nf, (PetscInt)massFractions->massFractionFunctions.size()); s++) {
            yi[s] = massFractions->massFractionFunctions[s] ? massFractions->massFractionFunctions[s]->Eval(x, dim, time) : 0.0;
            yiSum += yi[s];
        }

        for (PetscInt s = 0; s < PetscMin(Nf, (PetscInt)massFractions->massFractionFunctions.size()); s++) {
            yi[s] /= yiSum;
        }
    } catch (std::exception &exp) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", exp.what());
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::FieldFunction, ablate::finiteVolume::fieldFunctions::MassFractions, "initializes the yi field function variables based upon a the list of functions and eos",
         ARG(ablate::eos::EOS, "eos", "The eos with the list of species"), ARG(std::vector<ablate::mathFunctions::FieldFunction>, "values", "The list of mass fraction functions"));