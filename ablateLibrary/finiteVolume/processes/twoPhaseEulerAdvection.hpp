#ifndef ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP
#define ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP

#include <petsc.h>
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "process.hpp"

namespace ablate::finiteVolume::processes {

class TwoPhaseEulerAdvection : public Process {
   private:
    struct DecodeDataStruct {
        std::shared_ptr<ablate::eos::EOS> eosGas;
        std::shared_ptr<ablate::eos::EOS> eosLiquid;
        PetscReal ke;
        PetscReal e;
        PetscReal rho;
        PetscReal Yg;
        PetscReal Yl;
        PetscInt dim;
        PetscReal* vel;
    };

    const std::shared_ptr<eos::EOS> eosGas;
    const std::shared_ptr<eos::EOS> eosLiquid;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidGas;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid;

    static PetscErrorCode FormFunction(SNES snes, Vec x, Vec F, void* ctx);

   public:
    static PetscErrorCode UpdateAuxTemperatureField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                        void* ctx);

    static PetscErrorCode UpdateAuxPressureField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                     void* ctx);

    static PetscErrorCode UpdateAuxVelocityField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                     void* ctx);

    static PetscErrorCode UpdateAuxVolumeFractionField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues,
                                                           PetscScalar* auxField, void* ctx);

    TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                           std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidGas,
                           std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid);
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

   private:
    static PetscErrorCode CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[],
                                                           const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                           const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux,
                                                           void* ctx);
    static PetscErrorCode CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                        const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[],
                                                        const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux, void* ctx);

   public:
    static void DecodeTwoPhaseEulerState(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid, PetscInt dim, const PetscReal* conservedValues, PetscReal densityVF,
                                         const PetscReal* normal, PetscReal* density, PetscReal* densityG, PetscReal* densityL, PetscReal* normalVelocity, PetscReal* velocity,
                                         PetscReal* internalEnergy, PetscReal* internalEnergyG, PetscReal* internalEnergyL, PetscReal* aG, PetscReal* aL, PetscReal* MG, PetscReal* ML, PetscReal* p,
                                         PetscReal* alpha);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP
