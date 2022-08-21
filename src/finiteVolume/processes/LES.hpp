#ifndef ABLATELIBRARY_LES_H
#define ABLATELIBRARY_LES_H

#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

class LES : public FlowProcess {
   private:
    const std::shared_ptr<eos::EOS> eos;
    const std::shared_ptr<eos::transport::TransportModel> transportModel;

    // constant values
    inline const static PetscReal c_k = 0.094;
    inline const static PetscReal c_e = 1.048;
    inline const static PetscReal c_p = 1.040;

    /* store LES diffusion  data */
    struct DiffusionData {
        eos::ThermodynamicTemperatureFunction diffFunction;
        eos::ThermodynamicFunction computeTemperatureFunction;

        PetscInt numberSpecies;
        PetscInt numberEV;
    };
    DiffusionData diffusionData;

    PetscInt numberSpecies{};

   public:
    explicit LES(std::shared_ptr<eos::transport::TransportModel> transportModelIn);

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

   private:
    /**
     * This computes the momentum transfer for SGS model for rhoU
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LESMomentumFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the Energy transfer for SGS model for rhoE
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LESEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the species transfer for SGS model for density-YI
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"yi"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LESSpeciesFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);
    /**
     * This computes the EV transfer for SGS model for densityEV
     * f = "euler"
     * u = {"euler", "densityEV"}
     * a = {"ev"}
     * ctx = lesDiffusionData
     * @return
     */
    static PetscErrorCode LESevFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[],
                                    const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * static support function to compute the turbulent stress tensor
     * @param dim
     * @param mut
     * @param tau
     * @return
     */
    static PetscErrorCode CompressibleFlowComputeLESStressTensor(PetscInt dim, const PetscFVFaceGeom* fg, const PetscReal* gradVel, const PetscInt uOff[], const PetscScalar field[], void* ctx,
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
    static PetscErrorCode LESViscosity(PetscInt dim, const PetscFVFaceGeom* fg, const PetscScalar field[], const PetscInt uOff[], void* ctx, PetscReal& mut);
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_LES_H
