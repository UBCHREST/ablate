#ifndef ABLATELIBRARY_LES_H
#define ABLATELIBRARY_LES_H

#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

class LES : public FlowProcess {
   private:
    // store the conserved and non conserved form of the ev.

    const std::string ttke;
    //const std::string ;

    const std::shared_ptr<eos::EOS> eos;

PetscInt string;
    // constant values
    inline const static PetscReal c_k = 0.094;
    inline const static PetscReal c_e = 1.048;
    inline const static PetscReal c_p = 1.040;

    //PetscInt tke_ev;

    /* store turbulent diffusion  data */
    struct DiffusionData {

        eos::ThermodynamicFunction computeTemperatureFunction;

        PetscInt numberSpecies;
        PetscInt numberEV;

        PetscInt tke_ev;




    };
    DiffusionData diffusionData;
    PetscInt numberSpecies;
    PetscInt numberEV;



   public:
    explicit LES(  std::string ttke, std::shared_ptr<eos::EOS> eos);

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

   public:
    /**
     * This computes the momentum transfer for SGS model for rhoU
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LesMomentumFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the Energy transfer for SGS model for rhoE
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LesEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the species transfer for SGS model for density-YI
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LesSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the EV transfer for SGS model for densityEV
     * f = "euler"
     * u = {"euler", "densityEV"}
     * a = {"ev"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LesEvFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[],
                                    const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * static support function to compute the turbulent stress tensor
     * @param dim
     * @param mut
     * @param tau
     * @return
     */
    static PetscErrorCode CompressibleFlowComputeLesStressTensor(PetscInt dim, const PetscFVFaceGeom* fg, const PetscReal* gradVel, const PetscInt uOff[], const PetscScalar field[], void* ctx,
                                                                 PetscReal* turbTau);

    /**
     * static support function to compute the turbulent viscosity
     * @param dim
     * @param fg
     * @param field
     * @param uOff
     * @param mut
     * @return
     */
    static PetscErrorCode LesViscosity(PetscInt dim, void * ctx, const PetscFVFaceGeom* fg, const PetscScalar field[], const PetscInt uOff[], PetscReal& mut);
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_LES_H
