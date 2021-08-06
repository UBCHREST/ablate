#ifndef ABLATELIBRARY_SUTHERLAND_HPP
#define ABLATELIBRARY_SUTHERLAND_HPP

#include <eos/eos.hpp>
#include "transportModel.hpp"
namespace ablate::eos::transport {

class Sutherland : public TransportModel {
   private:
    const std::shared_ptr<eos::EOS> eos;
    const ComputeSpecificHeatConstantPressureFunction cpFunction;
    void* cpContext;

    // constant values
    inline const static PetscReal pr = 0.707;
    inline const static PetscReal muo=1.716e-5;
    inline const static PetscReal to=273.e+0;
    inline const static PetscReal so=111.e+0;
    inline const static PetscReal sc=0.707;

    static void SutherlandComputeConductivityFunction(PetscReal temperature, PetscReal density, const PetscReal* yi, PetscReal& conductivity, void* ctx);
    static void SutherlandComputeViscosityFunction(PetscReal temperature, PetscReal density, const PetscReal* yi, PetscReal& viscosity, void* ctx);
    static void SutherlandComputeDiffusivityFunction(PetscReal temperature, PetscReal density, const PetscReal* yi, PetscReal& diffusivity, void* ctx);

   public:
    explicit Sutherland(std::shared_ptr<eos::EOS> eos);

    ComputeConductivityFunction GetComputeConductivityFunction() override { return SutherlandComputeConductivityFunction; }
    void* GetComputeConductivityContext() override { return this; }
    ComputeViscosityFunction GetComputeViscosityFunction() override { return SutherlandComputeViscosityFunction; }
    void* GetComputeViscosityContext() override { return nullptr; }
    ComputeDiffusivityFunction GetComputeDiffusivityFunction() override { return SutherlandComputeDiffusivityFunction; }
    void* GetComputeDiffusivityContext() override { return nullptr; }
};
}  // namespace ablate::eos::transport

#endif  // ABLATELIBRARY_SUTHERLAND_HPP
