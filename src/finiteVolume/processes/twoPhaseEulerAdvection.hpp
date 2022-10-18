#ifndef ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP
#define ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP

#include <petsc.h>
#include "eos/perfectGas.hpp"
#include "eos/stiffenedGas.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "process.hpp"

namespace ablate::finiteVolume::processes {

class TwoPhaseEulerAdvection : public Process {
   private:
    struct DecodeDataStructGas {
        PetscReal etot;
        PetscReal rhotot;
        PetscReal Yg;
        PetscReal Yl;
        PetscReal gam1;
        PetscReal gam2;
        PetscReal cvg;
        PetscReal cpl;
        PetscReal p0l;
    };
    struct DecodeDataStructStiff {
        PetscReal etot;
        PetscReal rhotot;
        PetscReal Yg;
        PetscReal Yl;
        PetscReal gam1;
        PetscReal gam2;
        PetscReal cpg;
        PetscReal cpl;
        PetscReal p0g;
        PetscReal p0l;
    };
    static PetscErrorCode FormFunctionGas(SNES snes, Vec x, Vec F, void *ctx);
    static PetscErrorCode FormJacobianGas(SNES snes, Vec x, Mat J, Mat P, void *ctx);
    static PetscErrorCode FormFunctionStiff(SNES snes, Vec x, Vec F, void *ctx);
    static PetscErrorCode FormJacobianStiff(SNES snes, Vec x, Mat J, Mat P, void *ctx);

    PetscErrorCode MultiphaseFlowPreStage(TS flowTs, ablate::solver::Solver &flow, PetscReal stagetime);
    /**
     * General two phase decoder interface
     */
    class TwoPhaseDecoder {
       public:
        virtual void DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, PetscReal *densityG,
                                              PetscReal *densityL, PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *internalEnergyG, PetscReal *internalEnergyL,
                                              PetscReal *aG, PetscReal *aL, PetscReal *MG, PetscReal *ML, PetscReal *p, PetscReal *T, PetscReal *alpha) = 0;
        virtual ~TwoPhaseDecoder() = default;
    };

    /**
     * Implementation for two perfect gases
     */
    class PerfectGasPerfectGasDecoder : public TwoPhaseDecoder {
        const std::shared_ptr<eos::PerfectGas> eosGas;
        const std::shared_ptr<eos::PerfectGas> eosLiquid;

        /**
         * Store a scratch euler field for use with the eos
         */
        std::vector<PetscReal> gasEulerFieldScratch;
        std::vector<PetscReal> liquidEulerFieldScratch;

        /**
         * Get the compute functions using a fake field with only euler
         */
        eos::ThermodynamicFunction gasComputeTemperature;
        eos::ThermodynamicTemperatureFunction gasComputeInternalEnergy;
        eos::ThermodynamicTemperatureFunction gasComputeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction gasComputePressure;

        eos::ThermodynamicFunction liquidComputeTemperature;
        eos::ThermodynamicTemperatureFunction liquidComputeInternalEnergy;
        eos::ThermodynamicTemperatureFunction liquidComputeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction liquidComputePressure;

       public:
        PerfectGasPerfectGasDecoder(PetscInt dim, const std::shared_ptr<eos::PerfectGas> &perfectGasEos1, const std::shared_ptr<eos::PerfectGas> &perfectGasEos2);
        void DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, PetscReal *densityG, PetscReal *densityL,
                                      PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *internalEnergyG, PetscReal *internalEnergyL, PetscReal *aG, PetscReal *aL,
                                      PetscReal *MG, PetscReal *ML, PetscReal *p, PetscReal *T, PetscReal *alpha) override;
    };

    /**
     * Implementation for perfect gas and stiffened gas
     */
    class PerfectGasStiffenedGasDecoder : public TwoPhaseDecoder {
        const std::shared_ptr<eos::PerfectGas> eosGas;
        const std::shared_ptr<eos::StiffenedGas> eosLiquid;

        /**
         * Store a scratch euler field for use with the eos
         */
        std::vector<PetscReal> gasEulerFieldScratch;
        std::vector<PetscReal> liquidEulerFieldScratch;

        /**
         * Get the compute functions using a fake field with only euler
         */
        eos::ThermodynamicFunction gasComputeTemperature;
        eos::ThermodynamicTemperatureFunction gasComputeInternalEnergy;
        eos::ThermodynamicTemperatureFunction gasComputeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction gasComputePressure;

        eos::ThermodynamicFunction liquidComputeTemperature;
        eos::ThermodynamicTemperatureFunction liquidComputeInternalEnergy;
        eos::ThermodynamicTemperatureFunction liquidComputeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction liquidComputePressure;

       public:
        PerfectGasStiffenedGasDecoder(PetscInt dim, const std::shared_ptr<eos::PerfectGas> &perfectGasEos1, const std::shared_ptr<eos::StiffenedGas> &perfectGasEos2);

        void DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, PetscReal *densityG, PetscReal *densityL,
                                      PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *internalEnergyG, PetscReal *internalEnergyL, PetscReal *aG, PetscReal *aL,
                                      PetscReal *MG, PetscReal *ML, PetscReal *p, PetscReal *T, PetscReal *alpha) override;
    };

    /**
     * Implementation for two stiffened gases
     */
    class StiffenedGasStiffenedGasDecoder : public TwoPhaseDecoder {
        const std::shared_ptr<eos::StiffenedGas> eosGas;
        const std::shared_ptr<eos::StiffenedGas> eosLiquid;

        /**
         * Store a scratch euler field for use with the eos
         */
        std::vector<PetscReal> gasEulerFieldScratch;
        std::vector<PetscReal> liquidEulerFieldScratch;

        /**
         * Get the compute functions using a fake field with only euler
         */
        eos::ThermodynamicFunction gasComputeTemperature;
        eos::ThermodynamicTemperatureFunction gasComputeInternalEnergy;
        eos::ThermodynamicTemperatureFunction gasComputeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction gasComputePressure;

        eos::ThermodynamicFunction liquidComputeTemperature;
        eos::ThermodynamicTemperatureFunction liquidComputeInternalEnergy;
        eos::ThermodynamicTemperatureFunction liquidComputeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction liquidComputePressure;

       public:
        StiffenedGasStiffenedGasDecoder(PetscInt dim, const std::shared_ptr<eos::StiffenedGas> &perfectGasEos1, const std::shared_ptr<eos::StiffenedGas> &perfectGasEos2);

        void DecodeTwoPhaseEulerState(PetscInt dim, const PetscInt *uOff, const PetscReal *conservedValues, const PetscReal *normal, PetscReal *density, PetscReal *densityG, PetscReal *densityL,
                                      PetscReal *normalVelocity, PetscReal *velocity, PetscReal *internalEnergy, PetscReal *internalEnergyG, PetscReal *internalEnergyL, PetscReal *aG, PetscReal *aL,
                                      PetscReal *MG, PetscReal *ML, PetscReal *p, PetscReal *T, PetscReal *alpha) override;
    };

    const std::shared_ptr<eos::EOS> eosGas;
    const std::shared_ptr<eos::EOS> eosLiquid;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidGas;
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid;

    /**
     * Create and store the decoder
     */
    std::shared_ptr<TwoPhaseDecoder> decoder;

   public:
    static PetscErrorCode UpdateAuxTemperatureField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], const PetscScalar *conservedValues, const PetscInt aOff[],
                                                        PetscScalar *auxField, void *ctx);

    static PetscErrorCode UpdateAuxPressureField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], const PetscScalar *conservedValues, const PetscInt aOff[],
                                                     PetscScalar *auxField, void *ctx);

    static PetscErrorCode UpdateAuxVelocityField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], const PetscScalar *conservedValues, const PetscInt aOff[],
                                                     PetscScalar *auxField, void *ctx);

    static PetscErrorCode UpdateAuxVolumeFractionField2Gas(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt uOff[], const PetscScalar *conservedValues,
                                                           const PetscInt aOff[], PetscScalar *auxField, void *ctx);

    TwoPhaseEulerAdvection(std::shared_ptr<eos::EOS> eosGas, std::shared_ptr<eos::EOS> eosLiquid, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasGas,
                           std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorGasLiquid, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidGas,
                           std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculatorLiquidLiquid);
    void Setup(ablate::finiteVolume::FiniteVolumeSolver &flow) override;

   private:
    static PetscErrorCode CompressibleFlowComputeEulerFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                           const PetscInt aOff[], const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);
    static PetscErrorCode CompressibleFlowComputeVFFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar *flux, void *ctx);

   public:
    /**
     * static call to create a TwoPhaseDecoder based upon eos
     * @param dim
     * @param eosGas
     * @param eosLiquid
     * @return
     */
    static std::shared_ptr<TwoPhaseDecoder> CreateTwoPhaseDecoder(PetscInt dim, const std::shared_ptr<eos::EOS> &eosGas, const std::shared_ptr<eos::EOS> &eosLiquid);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_TWOPHASEEULERADVECTION_HPP
