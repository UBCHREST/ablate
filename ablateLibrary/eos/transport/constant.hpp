#ifndef ABLATELIBRARY_TRANSPORT_MODEL_CONSTANT_HPP
#define ABLATELIBRARY_TRANSPORT_MODEL_CONSTANT_HPP
#include "transportModel.hpp"
namespace ablate::eos::transport {

class Constant : public TransportModel {
   private:
    const bool active;
    const PetscReal k;
    const PetscReal mu;
    const PetscReal diff;

    static void ConstantFunction(PetscReal, PetscReal, const PetscReal* yi, PetscReal&, void* ctx);

   public:
    explicit Constant(double k = 0, double mu = 0, double diff = 0);
    explicit Constant(const Constant&) = delete;
    void operator=(const Constant&) = delete;

    ComputeConductivityFunction GetComputeConductivityFunction() override { return active ? ConstantFunction : nullptr; }
    void* GetComputeConductivityContext() override { return (void*)&k; }
    ComputeViscosityFunction GetComputeViscosityFunction() override { return active ? ConstantFunction : nullptr; }
    void* GetComputeViscosityContext() override { return (void*)&mu; }
    ComputeDiffusivityFunction GetComputeDiffusivityFunction() override { return active ? ConstantFunction : nullptr; }
    void* GetComputeDiffusivityContext() override { return (void*)&diff; }
};
}  // namespace ablate::eos::transport
#endif  // ABLATELIBRARY_CONSTANT_HPP
