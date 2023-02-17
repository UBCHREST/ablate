#ifndef ABLATELIBRARY_THERMOPHORETICDIFFUSION_HPP
#define ABLATELIBRARY_THERMOPHORETICDIFFUSION_HPP

#include "eos/transport/transportModel.hpp"
#include "finiteVolume/fluxCalculator/fluxCalculator.hpp"
#include "flowProcess.hpp"

namespace ablate::finiteVolume::processes {

/**
 * Thermophoretic diffusion the transport of ndd (ThermoPheretic) and solid carbon (ThermoPheretic).
 */
class ThermophoreticDiffusion : public FlowProcess {
   private:

    /**
     * Store the equation of state to compute pressure
     */
    const std::shared_ptr<eos::transport::TransportModel> transportModel;

    // store the thermodynamicTemperatureFunction to compute viscosity
    eos::ThermodynamicTemperatureFunction viscosityTemperatureFunction;

   public:
    explicit ThermophoreticDiffusion(std::shared_ptr<eos::transport::TransportModel> transportModel);

    /**
     * public function to link this process with the flow
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& flow) override;

   private:
    /**
     * This computes the energy transfer for species diffusion flux for rhoE
     * f = "euler"
     * u = {"euler", "densityYi"}
     * a = {"T"}
     * ctx = Viscosity Temperature Function
     * @return
     */
    static PetscErrorCode ThermophoreticDiffusionEnergyFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * This computes the species transfer for species diffusion flux
     * f = "densityYi" or "progressYi"
     * u = {"euler", "densityYi"} or {"euler", "progressYi"}
     * a = {"T"}
     * ctx = Viscosity Temperature Function
     * @return
     */
    static PetscErrorCode ThermophoreticDiffusionVariableFlux(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                               const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);



};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_SPECIESTRANSPORT_HPP
