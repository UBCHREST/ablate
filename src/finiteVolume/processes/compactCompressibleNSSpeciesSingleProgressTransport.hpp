#ifndef ABLATELIBRARY_COMPACTCOMPRESSIBLENSSPECIESSingleProgressTRANSPORT_H
#define ABLATELIBRARY_COMPACTCOMPRESSIBLENSSPECIESSingleProgressTRANSPORT_H

#include <petsc.h>
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "pressureGradientScaling.hpp"

namespace ablate::finiteVolume::processes {

class CompactCompressibleNSSpeciesSingleProgressTransport : public FlowProcess {
   private:
    // store ctx needed for function advection function that is passed into Petsc
    struct AdvectionData {
        // flow CFL
        PetscReal cfl;

        /* number of gas species and extra species */
        PetscInt numberSpecies;
        PetscInt numberEV;

        // EOS function calls (For Temperature it's better to just use the TemperatureTemperature function so we can guess Temperature from t
        eos::ThermodynamicTemperatureFunction computeTemperature;
        eos::ThermodynamicTemperatureFunction computeInternalEnergy;
        eos::ThermodynamicTemperatureFunction computeSpeedOfSound;
        eos::ThermodynamicTemperatureFunction computePressure;

        /* store method used for flux calculator */
        ablate::finiteVolume::fluxCalculator::FluxCalculatorFunction fluxCalculatorFunction;
        void* fluxCalculatorCtx;
    };
    AdvectionData advectionData;

    struct DiffusionData {
        /* thermal conductivity Diffusivity, and Dynamic Viscosity*/
        eos::ThermodynamicTemperatureFunction kFunction;
        eos::ThermodynamicTemperatureFunction muFunction;
        eos::ThermodynamicTemperatureFunction diffFunction;
        eos::ThermodynamicTemperatureFunction evDiffFunction;

        /* number of gas species and components in the SingleProgress */
        PetscInt numberSpecies;
        PetscInt numberEV;

        /* functions to compute species enthalpy */
        eos::ThermodynamicTemperatureFunction computeSpeciesSensibleEnthalpyFunction;
        /* store a scratch space for speciesSpeciesSensibleEnthalpy */
        std::vector<PetscReal> speciesSpeciesSensibleEnthalpy;
        /* store an optional scratch space for individual species diffusion */
        std::vector<PetscReal> speciesDiffusionCoefficient;
        /* store a scratch space for evDiffusionCoefficient */
        std::vector<PetscReal> evDiffusionCoefficient;
    };
    DiffusionData diffusionData;

    // Store the required ctx for time stepping
    struct CflTimeStepData {
        /* thermal conductivity*/
        AdvectionData* advectionData;
        /* pressure gradient scaling */
        std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> pgs;
    };
    CflTimeStepData timeStepData;

    //! methods and functions to compute diffusion based time stepping
    struct DiffusionTimeStepData {
        /* number of gas species */
        PetscInt numberSpecies;
        /* store an optional scratch space for individual species diffusion */
        std::vector<PetscReal> speciesDiffusionCoefficient;
        PetscReal SingleProgressDiffusionCoefficient;

        //! stability factor for condition time step. 0 (default) does not compute factor
        PetscReal diffusiveStabilityFactor;
        //! stability factor for condition time step. 0 (default) does not compute factor
        PetscReal conductionStabilityFactor;
        //! stability factor for viscous diffusion time step. 0 (default) does not compute factor
        PetscReal viscousStabilityFactor;

        /* diffusivity */
        eos::ThermodynamicTemperatureFunction diffFunction;
        /* thermal conductivity*/
        eos::ThermodynamicTemperatureFunction kFunction;
        /* dynamic viscosity*/
        eos::ThermodynamicTemperatureFunction muFunction;
        /* specific heat*/
        eos::ThermodynamicTemperatureFunction specificHeat;
        /* density */
        eos::ThermodynamicTemperatureFunction density;
    };
    DiffusionTimeStepData diffusionTimeStepData;

    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculator;
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<eos::transport::TransportModel> transportModel;
    const std::shared_ptr<eos::transport::TransportModel> SingleProgressTransportModel;

    eos::ThermodynamicTemperatureFunction computeTemperatureFunction;
    eos::ThermodynamicFunction computePressureFunction;

   public:
    explicit CompactCompressibleNSSpeciesSingleProgressTransport(const std::shared_ptr<parameters::Parameters>& parameters, std::shared_ptr<eos::EOS> eos,
                                                                 std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn = {}, std::shared_ptr<eos::transport::TransportModel> baseTransport = {},
                                                                 std::shared_ptr<eos::transport::TransportModel> evTransport = {},
                                                                 std::shared_ptr<ablate::finiteVolume::processes::PressureGradientScaling> = {});
    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * This Computes the Advective Flow for rho, rhoE, and rhoVel, rhoYi, and rhoEV.
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
     * This computes the energy transfer for species diffusion flux for rhoE
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * This computes the energy transfer for species diffusion flux for rhoE for variable diffusion coefficient
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionEnergyFluxVariableDiffusionCoefficient(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                          const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
                                                                          const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * This computes the species transfer for species diffusion flux
     * f = "densityYi"
     * u = {"euler"}
     * a = {"yi", "T"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * This computes the species transfer for species diffusion flux for variable diffusion coefficient
     * f = "densityYi"
     * u = {"euler"}
     * a = {"yi", "T"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionSpeciesFluxVariableDiffusionCoefficient(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                           const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
                                                                           const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the species transfer for species diffusion flux
     * f = "densityYi"
     * u = {"euler"}
     * a = {"yi", "T"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionEVFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * This computes the species transfer for species diffusion flux for variable diffusion coefficient
     * f = "densityYi"
     * u = {"euler"}
     * a = {"yi", "T"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionEVFluxVariableDiffusionCoefficient(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[],
                                                                      const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[],
                                                                      PetscScalar flux[], void* ctx);

    // static function to compute time step for euler advection
    static double ComputeCflTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx);
    // static function to compute the conduction based time step
    static double ComputeViscousDiffusionTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx);
    // static function to compute the conduction based time step
    static double ComputeConductionTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx);
    // static function to compute the conduction based time step
    static double ComputeViscousSpeciesDiffusionTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx);
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_COMPACTCOMPRESSIBLENSSPECIESSingleProgressTRANSPORT_H
