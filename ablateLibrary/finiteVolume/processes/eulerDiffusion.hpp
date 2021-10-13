#ifndef ABLATELIBRARY_EULERDIFFUSION_HPP
#define ABLATELIBRARY_EULERDIFFUSION_HPP
#include <eos/transport/transportModel.hpp>
#include "process.hpp"

namespace ablate::finiteVolume::processes {

class EulerDiffusion : public Process {
   public:
    struct _EulerDiffusionData {
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

        // EOS function calls
        PetscErrorCode (*computeTemperatureFunction)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal* densityYi, PetscReal* T, void* ctx);
        void* computeTemperatureContext;
    };
    typedef struct _EulerDiffusionData* EulerDiffusionData;

    explicit EulerDiffusion(std::shared_ptr<eos::EOS> eos, std::shared_ptr<eos::transport::TransportModel> transportModel);
    ~EulerDiffusion() override;

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Initialize(ablate::finiteVolume::FVFlow& flow) override;

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
    std::shared_ptr<eos::transport::TransportModel> transportModel;
    /**
     * This Computes the diffusion flux for euler rhoE, rhoVel
     * u = {"euler", "densityYi"}
     * a = {"temperature", "velocity"}
     * ctx = FlowData_CompressibleFlow
     * @return
     */
    static PetscErrorCode CompressibleFlowEulerDiffusion(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar fieldL[],
                                                         const PetscScalar fieldR[], const PetscScalar gradL[], const PetscScalar gradR[], const PetscInt aOff[], const PetscInt aOff_x[],
                                                         const PetscScalar auxL[], const PetscScalar auxR[], const PetscScalar gradAuxL[], const PetscScalar gradAuxR[], PetscScalar* fL, void* ctx);
    /**
     * Function to compute the temperature field. This function assumes that the input values will be {"euler", "densityYi"}
     */
    static PetscErrorCode UpdateAuxTemperatureField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                    void* ctx);
    /**
     * Function to compute the velocity. This function assumes that the input values will be {"euler"}
     */
    static PetscErrorCode UpdateAuxVelocityField(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* conservedValues, PetscScalar* auxField,
                                                 void* ctx);
};

}  // namespace ablate::flow::processes
#endif  // ABLATELIBRARY_EULERDIFFUSION_HPP
