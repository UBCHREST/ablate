#ifndef ABLATELIBRARY_SPECIESDIFFUSION_HPP
#define ABLATELIBRARY_SPECIESDIFFUSION_HPP

#include "flowProcess.hpp"

namespace ablate::flow::processes {

class SpeciesDiffusion : public FlowProcess {
   public:
    struct _SpeciesDiffusionData {
        /* diffusivity */
        PetscReal diff;

        /* number of gas species */
        PetscInt numberSpecies;
    };
    typedef struct _SpeciesDiffusionData* SpeciesDiffusionData;

    explicit SpeciesDiffusion(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eos);
    ~SpeciesDiffusion() override;

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::flow::FVFlow& flow) override;

   private:
    SpeciesDiffusionData speciesDiffusionData;
    std::shared_ptr<eos::EOS> eos;

    /**
     * Function to compute the mass fraction. This function assumes that the input values will be {"euler", "densityYi"}
     */
    static PetscErrorCode UpdateAuxMassFractionField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                     void* ctx);

    /**
     * This computes the energy transfer for species diffusion flux for rhoE
     * f = "euler"
     * u = {"euler"}
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
     * a = {"yi"}
     * ctx = SpeciesDiffusionData
     * @return
     */
    static PetscErrorCode SpeciesDiffusionSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[], const PetscScalar fieldR[],
                                                      const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar auxL[],
                                                      const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);
};

}  // namespace ablate::flow::processes
#endif  // ABLATELIBRARY_SPECIESDIFFUSION_HPP
