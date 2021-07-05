#ifndef ABLATELIBRARY_EULERDIFFUSION_HPP
#define ABLATELIBRARY_EULERDIFFUSION_HPP
#include "flowProcess.hpp"

namespace ablate::flow::processes {

class EulerDiffusion : public FlowProcess {
    typedef enum { T, VEL, TOTAL_COMPRESSIBLE_AUX_COMPONENTS } CompressibleAuxComponents;

   public:
    struct _EulerDiffusionData {
        /* thermal conductivity*/
        PetscReal k;
        /* dynamic viscosity*/
        PetscReal mu;
        /* number of gas species */
        PetscInt numberSpecies;

        // EOS function calls
        PetscErrorCode (*computeTemperatureFunction)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal* densityYi, PetscReal* T, void* ctx);
        void* computeTemperatureContext;
    };
    typedef struct _EulerDiffusionData* EulerDiffusionData;

    explicit EulerDiffusion(std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<eos::EOS> eos);
    ~EulerDiffusion() override;

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::flow::FVFlow& flow) override;

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

   private:
    EulerDiffusionData eulerDiffusionData;
    std::shared_ptr<eos::EOS> eos;

    /**
     * This Computes the diffusion flux for euler rhoE, rhoVel
     * u = {"euler"}
     * a = {"temperature", "velocity"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[],
                                                         const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                         const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);
    // function to update the aux temperature field
    static PetscErrorCode UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx);
    static PetscErrorCode UpdateAuxVelocityField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscScalar* conservedValues, PetscScalar* auxField, void* ctx);
};

}  // namespace ablate::flow::processes
#endif  // ABLATELIBRARY_EULERDIFFUSION_HPP
