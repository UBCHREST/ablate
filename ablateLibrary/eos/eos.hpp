#ifndef ABLATELIBRARY_EOS_HPP
#define ABLATELIBRARY_EOS_HPP
#include <petsc.h>
#include <iostream>
#include <string>
#include <vector>

namespace ablate::eos {

using DecodeStateFunction = PetscErrorCode (*)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, const PetscReal densityYi[], PetscReal* internalEnergy, PetscReal* a,
                                               PetscReal* p, void* ctx);
using ComputeTemperatureFunction = PetscErrorCode (*)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx);

using ComputeReactionRateFunction = PetscErrorCode (*)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal *densityEnergySource, PetscReal* densityYiSource, void* ctx);

/**
 * The EOS is a combination of species model and EOS.  This allows the eos to dictate the order/number of species.  This can be relaxed in the future
 */
class EOS {
   protected:
    const std::string type;

   public:
    EOS(std::string typeIn) : type(typeIn){};
    virtual ~EOS() = default;

    // Print the details of this eos
    virtual void View(std::ostream& stream) const = 0;

    // eos functions are accessed through getting the function directly
    virtual DecodeStateFunction GetDecodeStateFunction() = 0;
    virtual void* GetDecodeStateContext() = 0;
    virtual ComputeTemperatureFunction GetComputeTemperatureFunction() = 0;
    virtual void* GetComputeTemperatureContext() = 0;
    virtual ComputeReactionRateFunction GetComputeReactionRateFunction() = 0;
    virtual void* GetComputeReactionRateContext() = 0;

    // species model functions
    virtual const std::vector<std::string>& GetSpecies() const = 0;

    // Support function for printing any eos
    friend std::ostream& operator<<(std::ostream& out, const EOS& eos) {
        eos.View(out);
        return out;
    }
};
}  // namespace ablate::eos

#endif  // ABLATELIBRARY_EOS_HPP
