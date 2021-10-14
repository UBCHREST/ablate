#ifndef ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP
#define ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP

#include <petsc.h>
#include "flow/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"

namespace ablate::flow::processes {

class TwoPhaseEulerAdvection : public FlowProcess {
   private:
    const std::shared_ptr<eos::EOS> eosGas;
    const std::shared_ptr<eos::EOS> eosLiquid;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid;
   public:
    TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid );
    void Initialize(ablate::flow::FVFlow& flow) override;
   private:
    static PetscErrorCode CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[],
                                                           const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                           const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux,
                                                           void* ctx);
    static PetscErrorCode CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[],
                                                           const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                           const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux,
                                                           void* ctx);
   public:
    static void DecodeTwoPhaseEulerState(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid, PetscInt dim, const PetscReal* conservedValues,
                                         const PetscReal* normal, PetscReal* density, PetscReal* densityG, PetscReal* densityL, PetscReal* normalVelocity, PetscReal* velocity, PetscReal* internalEnergy, PetscReal* internalEnergyG, PetscReal* internalEnergyL, PetscReal* aG, PetscReal* aL, PetscReal* MG, PetscReal* ML, PetscReal* p, PetscReal* alpha, void *ctx);


    //           *** yaml example ***
//      - !ablate::flow::processes::TwoPhaseEulerAdvection
    //      parameters:
    //        cfl: 0.5
    //      eosGas: !ablate::eos::PerfectGas
    //        parameters:
    //          gamma: 1.4
    //          Rgas : 287.0 
    //      eosLiquid: !ablate::eos::StiffenedGas
    //        parameters:
    //          gamma: 1.4
    //          Cv : 287.0
    //          p0 : 1.0
    //          T0 : 1.0
    //          e0 : 1.0 
    //      fluxCalculatorGasGas: !ablate::flow::fluxCalculator::AusmpUp
    //        mInf: .3
    //      fluxCalculatorGasLiquid: !ablate::flow::fluxCalculator::AusmpUp
    //        mInf: .3
    //      fluxCalculatorLiquidLiquid: !ablate::flow::fluxCalculator::AusmpUp
    //        mInf: .3
};

}  // namespace ablate::flow::processes
#endif  // ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP
