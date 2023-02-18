#ifndef ABLATELIBRARY_SPECIESTRANSPORT_HPP
#define ABLATELIBRARY_SPECIESTRANSPORT_HPP

#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

class SpeciesTransport : public FlowProcess {
   private:
    const std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalculator;
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<eos::transport::TransportModel> transportModel;

    // Store ctx needed for static function advection function passed to PETSc
    struct AdvectionData {
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
    AdvectionData advectionData;

    struct DiffusionData {
        /* diffusivity */
        eos::ThermodynamicTemperatureFunction diffFunction;

        /* number of gas species */
        PetscInt numberSpecies;

        /* functions to compute species enthalpy */
        eos::ThermodynamicFunction computeTemperatureFunction;
        eos::ThermodynamicTemperatureFunction computeSpeciesSensibleEnthalpyFunction;

        /* store a scratch space for speciesSpeciesSensibleEnthalpy */
        std::vector<PetscReal> speciesSpeciesSensibleEnthalpy;
        /* store an optional scratch space for individual species diffusion */
        std::vector<PetscReal> speciesDiffusionCoefficient;
    };
    DiffusionData diffusionData;

    //! methods and functions to compute diffusion based time stepping
    struct DiffusionTimeStepData {
        /* number of gas species */
        PetscInt numberSpecies;

        //! stability factor for condition time step. 0 (default) does not compute factor
        PetscReal stabilityFactor;

        /* diffusivity */
        eos::ThermodynamicTemperatureFunction diffFunction;
        /* store an optional scratch space for individual species diffusion */
        std::vector<PetscReal> speciesDiffusionCoefficient;
    };
    DiffusionTimeStepData diffusionTimeStepData;

    // Store ctx needed for static function diffusion function passed to PETSc
    PetscInt numberSpecies;

   public:
    explicit SpeciesTransport(std::shared_ptr<eos::EOS> eos, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn = {}, std::shared_ptr<eos::transport::TransportModel> transportModel = {},
                              const std::shared_ptr<parameters::Parameters>& parametersIn = {});

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * Function to compute the mass fraction. This function assumes that the input values will be {"euler", "densityYi"}
     */
    static PetscErrorCode UpdateAuxMassFractionField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, const PetscInt aOff[],
                                                     PetscScalar* auxField, void* ctx);

    /**
     * Normalize and cleanup the species mass fractions in the solution vector
     * @param ts
     */
    static void NormalizeSpecies(TS ts, ablate::solver::Solver&);

   private:
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
     * This Computes the advection flux for each species (Yi)
     * u = {"euler", "densityYi"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode AdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscScalar fieldL[], const PetscScalar fieldR[], const PetscInt aOff[],
                                        const PetscScalar auxL[], const PetscScalar auxR[], PetscScalar* flux, void* ctx);

    // static function to compute the conduction based time step
    static double ComputeViscousDiffusionTimeStep(TS ts, ablate::finiteVolume::FiniteVolumeSolver& flow, void* ctx);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_SPECIESTRANSPORT_HPP
