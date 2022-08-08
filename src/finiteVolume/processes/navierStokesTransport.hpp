#ifndef ABLATELIBRARY_NAVIERSTOKESTRANSPORT_HPP
#define ABLATELIBRARY_NAVIERSTOKESTRANSPORT_HPP

#include <petsc.h>
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "pressureGradientScaling.hpp"

namespace ablate::finiteVolume::processes {

class NavierStokesTransport : public FlowProcess {
   public:
    // Store ctx needed for static function advection function passed to PETSc
    struct AdvectionData {
        // flow CFL
        PetscReal cfl;

        /* number of gas species */
        PetscInt numberSpecies;

        // EOS function calls
        eos::ThermodynamicFunction computeTemperature;
        eos::ThermodynamicTemperatureFunction computeInternalEnergy;
        eos::ThermodynamicTemperatureFunction computeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction computePressure;

        /* store method used for flux calculator */
        ablate::finiteVolume::fluxCalculator::FluxCalculatorFunction fluxCalculatorFunction;
        void* fluxCalculatorCtx;
    };

    // Store ctx needed for static function diffusion function passed to PETSc
    struct DiffusionData {
        /* thermal conductivity*/
        eos::ThermodynamicTemperatureFunction kFunction;
        /* dynamic viscosity*/
        eos::ThermodynamicTemperatureFunction muFunction;
    };

   private:
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculator;
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<eos::transport::TransportModel> transportModel;
    AdvectionData advectionData;

    eos::ThermodynamicTemperatureFunction computeTemperatureFunction;

    DiffusionData diffusionData;

    eos::ThermodynamicFunction computePressureFunction;

    // Store the required ctx for time stepping
    struct TimeStepData {
        /* thermal conductivity*/
        AdvectionData* advectionData;

        /**
         * pressure gradient scaling
         */
        std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs;
    };
    TimeStepData timeStepData;

    // static function to compute time step for euler advection
    static double ComputeTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx);

   public:
    /**
     * Function to compute the temperature field. This function assumes that the input values will be {"euler", "densityYi"}
     */
    static PetscErrorCode UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, const PetscInt aOff[],
                                                    PetscScalar* auxField, void* ctx);
    /**
     * Function to compute the velocity. This function assumes that the input values will be {"euler"}
     */
    static PetscErrorCode UpdateAuxVelocityField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, const PetscInt aOff[],
                                                 PetscScalar* auxField, void* ctx);

    /**
     * Function to compute the velocity. This function assumes that the input values will be {"euler", "densityYi }
     */
    static PetscErrorCode UpdateAuxPressureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, const PetscInt aOff[],
                                                 PetscScalar* auxField, void* ctx);

    /**
     *
     * public constructor for euler advection
     */
    NavierStokesTransport(const std::shared_ptr<parameters::Parameters>& parameters, std::shared_ptr<eos::EOS> eos, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn = {},
                          std::shared_ptr<eos::transport::TransportModel> transportModel = {}, std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> = {});

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * This Computes the Flow Euler flow for rho, rhoE, and rhoVel.
     * u = {"euler"} or {"euler", "densityYi"} if species are tracked
     * a = {}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode AdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar* flux, void* ctx);

    /**
     * This Computes the diffusion flux for euler rhoE, rhoVel
     * u = {"euler", "densityYi"}
     * a = {"temperature", "velocity"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode DiffusionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * static support function to compute the stress tensor
     * @param dim
     * @param mu
     * @param gradVelL
     * @param gradVelR
     * @param tau
     * @return
     */
    static PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVel, PetscReal* tau);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_NAVIERSTOKESTRANSPORT_HPP
