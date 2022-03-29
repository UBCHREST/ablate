#ifndef ABLATELIBRARY_EULERTRANSPORT_HPP
#define ABLATELIBRARY_EULERTRANSPORT_HPP

#include <petsc.h>
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

class EulerTransport : public FlowProcess {
   public:
    // Store ctx needed for static function advection function passed to PETSc
    struct AdvectionData {
        // flow CFL
        PetscReal cfl;

        /* number of gas species */
        PetscInt numberSpecies;

        // EOS function calls
        eos::DecodeStateFunction decodeStateFunction;
        void* decodeStateContext;

        /* store method used for flux calculator */
        ablate::finiteVolume::fluxCalculator::FluxCalculatorFunction fluxCalculatorFunction;
        void* fluxCalculatorCtx;
    };

    // Store ctx needed for static function diffusion function passed to PETSc
    struct DiffusionData {
        /* thermal conductivity*/
        eos::transport::ComputeConductivityFunction kFunction;
        void* kContext;
        /* dynamic viscosity*/
        eos::transport::ComputeViscosityFunction muFunction;
        void* muContext;

        /* store a scratch variable to hold yi*/
        std::vector<PetscReal> yiScratch;

        /* number of gas species */
        PetscInt numberSpecies;
    };
    // Store ctx needed for static function diffusion function passed to PETSc
    struct UpdateTemperatureData {
        eos::ComputeTemperatureFunction computeTemperatureFunction;
        void* computeTemperatureContext;
        PetscInt numberSpecies;
    };

   private:
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculator;
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<eos::transport::TransportModel> transportModel;
    AdvectionData advectionData;

    UpdateTemperatureData updateTemperatureData;

    DiffusionData diffusionData;

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
    EulerTransport(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eos, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn = {},
                   std::shared_ptr<eos::transport::TransportModel> transportModel = {});

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * This Computes the Flow Euler flow for rho, rhoE, and rhoVel.
     * u = {"euler"} or {"euler", "densityYi"} if species are tracked
     * a = {}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode AdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                        const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[],
                                        const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux, void* ctx);

    /**
     * This Computes the diffusion flux for euler rhoE, rhoVel
     * u = {"euler", "densityYi"}
     * a = {"temperature", "velocity"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode DiffusionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                        const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[],
                                        const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);

    /**
     * static support function to compute the stress tensor
     * @param dim
     * @param mu
     * @param gradVelL
     * @param gradVelR
     * @param tau
     * @return
     */
    static PetscErrorCode CompressibleFlowComputeStressTensor(PetscInt dim, PetscReal mu, const PetscReal* gradVelL, const PetscReal* gradVelR, PetscReal* tau);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_EULERTRANSPORT_HPP
