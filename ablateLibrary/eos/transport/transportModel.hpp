#ifndef ABLATELIBRARY_TRANSPORTMODEL_HPP
#define ABLATELIBRARY_TRANSPORTMODEL_HPP
#include <petscsystypes.h>
namespace ablate::eos::transport {

using ComputeConductivityFunction = void (*)(PetscReal temperature, PetscReal& conductivity,void* ctx);
using ComputeViscosityFunction = void (*)(PetscReal temperature, PetscReal& viscosity, void* ctx);
using ComputeDiffusivityFunction = void (*)(PetscReal temperature, PetscReal density, PetscReal& diffusivity, void* ctx);

class TransportModel {
   public:
    virtual ~TransportModel() = default;

    virtual ComputeConductivityFunction GetComputeConductivityFunction() = 0;
    virtual void* GetComputeConductivityContext() = 0;
    virtual ComputeViscosityFunction GetComputeViscosityFunction() = 0;
    virtual void* GetComputeViscosityContext() = 0;
    virtual ComputeDiffusivityFunction GetComputeDiffusivityFunction() = 0;
    virtual void* GetComputeDiffusivityContext() = 0;
};
}  // namespace ablate::eos::transport
#endif  // ABLATELIBRARY_TRANSPORTMODEL_HPP
