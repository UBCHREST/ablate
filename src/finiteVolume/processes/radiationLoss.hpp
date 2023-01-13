#ifndef ABLATELIBRARY_RADIATIONLOSS_HPP
#define ABLATELIBRARY_RADIATIONLOSS_HPP

#include "eos/radiationProperties/radiationProperties.hpp"
#include "process.hpp"
#include "utilities/constants.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

namespace ablate::finiteVolume::processes {
/**
 * This class uses math functions to add arbitrary sources to the fvm method
 */
class RadiationLoss : public Process {
    //! list of functions used to compute the arbitrary source
    const std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions;

    /**
     * private function to compute  source
     * @return
     */
    static PetscErrorCode ComputeRadiationLoss(PetscInt dim, PetscReal time, const PetscFVCellGeom* cg, const PetscInt uOff[], const PetscScalar u[], const PetscInt aOff[], const PetscScalar a[],
                                                 PetscScalar f[], void* ctx);

   public:
    explicit RadiationLoss(std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModelIn, double tInfinityIn = 300);

    /**
     * public function to link this process with the fvm solver
     * @param flow
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& fvmSolver) override;

    inline std::shared_ptr<eos::radiationProperties::RadiationModel> GetRadiationModel() { return radiationModel; }

    /**
     * Get the radiation losses from the cell in question based on the temperature and absorption properties
     * @param temperature
     * @param kappa
     * @return
     */
    static inline PetscReal GetIntensity(PetscReal tInfinity, PetscReal temperature, PetscReal kappa) {

        tInfinity = PetscMax(tInfinity, (temperature - 500));

        // Compute the losses
        PetscReal netIntensity = -4.0 * ablate::utilities::Constants::sbc * temperature * temperature * temperature * temperature;

        netIntensity += 4.0 * ablate::utilities::Constants::sbc * tInfinity * tInfinity * tInfinity * tInfinity;

        // scale by kappa
        netIntensity *= kappa;

        return PetscAbsReal(netIntensity) > ablate::utilities::Constants::large ? ablate::utilities::Constants::large * PetscSignReal(netIntensity) : netIntensity;
    }

    //! model used to provided the absorptivity function
    const std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel;

    //! hold a pointer to the absorptivity function
    eos::ThermodynamicTemperatureFunction absorptivityFunction;

    PetscReal tInfinity;
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_RADIATIONLOSS_HPP
