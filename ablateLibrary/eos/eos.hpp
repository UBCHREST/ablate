#ifndef ABLATELIBRARY_EOS_HPP
#define ABLATELIBRARY_EOS_HPP
#include <petsc.h>
#include <iostream>

namespace ablate::eos {

using decodeStateFunction = PetscErrorCode (*)(const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a,
                                               PetscReal* p, void* ctx);
using computeTemperatureFunction = PetscErrorCode (*)(const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, PetscReal* T, void* ctx);

class EOS {
   protected:
    const std::string type;
   public:
    EOS(std::string typeIn): type(typeIn){};
    virtual ~EOS() = default;

    // Print the details of this eos
    virtual void View(std::ostream& stream) const = 0;

    // eos functions are accessed through getting the function directly
    virtual decodeStateFunction GetDecodeStateFunction() = 0;
    virtual void* GetDecodeStateContext() = 0;
    virtual computeTemperatureFunction GetComputeTemperatureFunction() = 0;
    virtual void* GetComputeTemperatureContext() = 0;

    // Support function for printing any eos
    friend std::ostream& operator<<(std::ostream& out, const EOS& eos) {
        eos.View(out);
        return out;
    }
};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_EOS_HPP
