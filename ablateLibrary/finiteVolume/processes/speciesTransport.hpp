#ifndef ABLATELIBRARY_SPECIESTRANSPORT_HPP
#define ABLATELIBRARY_SPECIESTRANSPORT_HPP

#include <eos/transport/transportModel.hpp>
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
        eos::DecodeStateFunction decodeStateFunction;
        void* decodeStateContext;

        /* store method used for flux calculator */
        ablate::finiteVolume::fluxCalculator::FluxCalculatorFunction fluxCalculatorFunction;
        void* fluxCalculatorCtx;
    };
    AdvectionData advectionData;

    struct DiffusionData {
        /* diffusivity */
        eos::transport::ComputeDiffusivityFunction diffFunction;
        void* diffContext;

        /* number of gas species */
        PetscInt numberSpecies;

        /* functions to compute species enthalpy */
        eos::ComputeTemperatureFunction computeTemperatureFunction;
        void* computeTemperatureContext;
        eos::ComputeSpeciesSensibleEnthalpyFunction computeSpeciesSensibleEnthalpyFunction;
        void* computeSpeciesSensibleEnthalpyContext;

        /* store a scratch space for speciesSpeciesSensibleEnthalpy */
        std::vector<PetscReal> speciesSpeciesSensibleEnthalpy;
    };
    DiffusionData diffusionData;

    // Store ctx needed for static function diffusion function passed to PETSc
    PetscInt numberSpecies;

   public:
    explicit SpeciesTransport(std::shared_ptr<eos::EOS> eos, std::shared_ptr<fluxCalculator::FluxCalculator> fluxCalcIn = {}, std::shared_ptr<eos::transport::TransportModel> transportModel = {});

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

    /**
     * Function to compute the mass fraction. This function assumes that the input values will be {"euler", "densityYi"}
     */
    static PetscErrorCode UpdateAuxMassFractionField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                     void* ctx);

   private:
    /**
     * This computes the energy transfer for species diffusion flux for rhoE
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                              const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[],
                                              const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);
    /**
     * This computes the species transfer for species diffusion fluxy
     * f = "densityYi"
     * u = {"euler"}
     * a = {"yi", "T"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode DiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                               const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[],
                                               const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);

    /**
     * This Computes the advection flux for each species (Yi)
     * u = {"euler", "densityYi"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode AdvectionFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                        const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[], const PetscScalar auxR[],
                                        const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* flux, void* ctx);
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_SPECIESTRANSPORT_HPP
