#include "twoPhaseEulerAdvection.hpp"
ablate::flow::processes::TwoPhaseEulerAdvection::TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid,
                                                                        std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                                                                        std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid,
                                                                        std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid)
    : eosGas(eosGas), eosLiquid(eosLiquid), fluxCalculatorGasGas(fluxCalculatorGasGas), fluxCalculatorGasLiquid(fluxCalculatorGasLiquid), fluxCalculatorLiquidLiquid(fluxCalculatorLiquidLiquid) {}
void ablate::flow::processes::TwoPhaseEulerAdvection::Initialize(ablate::flow::FVFlow& flow) {
    flow.RegisterRHSFunction(CompressibleFlowComputeEulerFlux, this, "euler", {"densityVF","euler"},{});
    flow.RegisterRHSFunction(CompressibleFlowComputeVFFlux, this, "densityVF", {"densityVF","euler"},{});
}
PetscErrorCode ablate::flow::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x,
                                                                                                 const PetscScalar *fieldL, const PetscScalar *fieldR, const PetscScalar *gradL,
                                                                                                 const PetscScalar *gradR, const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *auxL,
                                                                                                 const PetscScalar *auxR, const PetscScalar *gradAuxL, const PetscScalar *gradAuxR, PetscScalar *flux,
                                                                                                 void *ctx) {
    PetscFunctionBeginUser;
//    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection*)ctx;

    PetscFunctionReturn(0);
}
PetscErrorCode ablate::flow::processes::TwoPhaseEulerAdvection::CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x,
                                                                                              const PetscScalar *fieldL, const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR,
                                                                                              const PetscInt *aOff, const PetscInt *aOff_x, const PetscScalar *auxL, const PetscScalar *auxR,
                                                                                              const PetscScalar *gradAuxL, const PetscScalar *gradAuxR, PetscScalar *flux, void *ctx) {
    PetscFunctionBeginUser;
//    auto twoPhaseEulerAdvection = (TwoPhaseEulerAdvection*)ctx;

    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::flow::processes::FlowProcess, ablate::flow::processes::TwoPhaseEulerAdvection, "",
         ARG(ablate::eos::EOS,"eosGas",""),ARG(ablate::eos::EOS,"eosLiquid",""),
         ARG(ablate::flow::fluxCalculator::FluxCalculator,"fluxCalculatorGasGas",""),ARG(ablate::flow::fluxCalculator::FluxCalculator,"fluxCalculatorGasLiquid",""),ARG(ablate::flow::fluxCalculator::FluxCalculator,"fluxCalculatorLiquidLiquid",""));
