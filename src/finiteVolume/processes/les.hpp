#ifndef ABLATELIBRARY_LES_H
#define ABLATELIBRARY_LES_H

#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"
#include "navierStokesTransport.hpp"


namespace ablate::finiteVolume::processes {

class LES : public FlowProcess {
   private:
    const std::string tke;

    // constant values
    inline const static PetscReal c_k = 0.094;
    inline const static PetscReal c_e = 1.048;
    inline const static PetscReal c_p = 1.040;
    inline const static PetscReal scT = 1.00;
    inline const static PetscReal prT= 1.00;

    /* store turbulent diffusion  data */
    struct DiffusionData {
        NavierStokesTransport *computeTau;
        PetscInt numberSpecies;
        PetscInt numberEV;
        PetscInt tke_ev;

    };
    DiffusionData diffusionData;
    PetscInt numberSpecies;


   public:
    explicit LES(std::string tke);

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

    /**
     * static support function to compute the turbulent viscosity
     * @param dim
     * @param fg
     * @param field
     * @param uOff
     * @param mut
     * @return
     */
    static PetscErrorCode LesViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar* densityField, const PetscReal turbulence, PetscReal& mut);
};
}// namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_LES_H
