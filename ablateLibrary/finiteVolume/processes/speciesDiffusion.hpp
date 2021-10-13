#ifndef ABLATELIBRARY_SPECIESDIFFUSION_HPP
#define ABLATELIBRARY_SPECIESDIFFUSION_HPP

#include <eos/transport/transportModel.hpp>
#include "process.hpp"

namespace ablate::finiteVolume::processes {

class SpeciesDiffusion : public Process {
   public:
    struct _SpeciesDiffusionData {
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
    typedef struct _SpeciesDiffusionData* SpeciesDiffusionData;

    explicit SpeciesDiffusion(std::shared_ptr<eos::EOS> eos, std::shared_ptr<eos::transport::TransportModel> transportModel);
    ~SpeciesDiffusion() override;

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FVFlow& flow) override;

   private:
    SpeciesDiffusionData speciesDiffusionData;
    std::shared_ptr<eos::EOS> eos;
    std::shared_ptr<eos::transport::TransportModel> transportModel;

    /**
     * Function to compute the mass fraction. This function assumes that the input values will be {"euler", "densityYi"}
     */
    static PetscErrorCode UpdateAuxMassFractionField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                     void* ctx);

    /**
     * This computes the energy transfer for species diffusion flux for rhoE
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode SpeciesDiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                     const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[],
                                                     const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);
    /**
     * This computes the species transfer for species diffusion fluxy
     * f = "densityYi"
     * u = {"euler"}
     * a = {"yi", "T"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode SpeciesDiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                      const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[],
                                                      const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);
};

}  // namespace ablate::flow::processes
#endif  // ABLATELIBRARY_SPECIESDIFFUSION_HPP
