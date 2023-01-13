#ifndef ABLATELIBRARY_RADIATIONLOSS_HPP
#define ABLATELIBRARY_RADIATIONLOSS_HPP

#include "eos/radiationProperties/radiationProperties.hpp"
#include "process.hpp"
#include "utilities/constants.hpp"
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

    //! pre store the petsc function and context
    struct PetscFunctionStruct {
        mathFunctions::PetscFunction petscFunction;
        void* petscContext;
        PetscInt fieldSize;
    };

    //! Store pointers to the petsc functions
    std::vector<PetscFunctionStruct> petscFunctions;

   public:
    explicit RadiationLoss(std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions);

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
    static inline PetscReal GetIntensity(PetscReal temperature, PetscReal kappa) {
        // Compute the losses
        PetscReal netIntensity = -4.0 * ablate::utilities::Constants::sbc * temperature * temperature * temperature * temperature;

        // scale by kappa
        netIntensity *= kappa;

        return abs(netIntensity) > ablate::utilities::Constants::large ? ablate::utilities::Constants::large * PetscSignReal(netIntensity) : netIntensity;
    }

    //! model used to provided the absorptivity function
    const std::shared_ptr<eos::radiationProperties::RadiationModel> radiationModel;

    //! hold a pointer to the absorptivity function
    static eos::ThermodynamicTemperatureFunction absorptivityFunction;
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_RADIATIONLOSS_HPP
