#ifndef ABLATELIBRARY_PERFECTGAS_HPP
#define ABLATELIBRARY_PERFECTGAS_HPP
#include <memory>
#include "eos.hpp"
#include "parameters/parameters.hpp"
namespace ablate::eos {

class PerfectGas : public EOS {
   private:
    // the perfect gas does not allow species
    const std::vector<std::string> species = {};
    struct Parameters {
        PetscReal gamma;
        PetscReal rGas;
    };
    Parameters parameters;

    static PetscErrorCode PerfectGasDecodeState(const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* velocity, PetscReal* internalEnergy, PetscReal* a,
                                                PetscReal* p, void* ctx);
    static PetscErrorCode PerfectGasComputeTemperature(const PetscReal* yi, PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, PetscReal* T, void* ctx);

   public:
    explicit PerfectGas(std::shared_ptr<ablate::parameters::Parameters>);
    void View(std::ostream& stream) const override;
    decodeStateFunction GetDecodeStateFunction() override { return PerfectGasDecodeState; }
    void* GetDecodeStateContext() override { return &parameters; }
    computeTemperatureFunction GetComputeTemperatureFunction() override { return PerfectGasComputeTemperature; }
    void* GetComputeTemperatureContext() override { return &parameters; }

    const std::vector<std::string>& GetSpecies() const override{
        return species;
    }
};

}  // namespace ablate::eos
#endif  // ABLATELIBRARY_PERFECTGAS_HPP
