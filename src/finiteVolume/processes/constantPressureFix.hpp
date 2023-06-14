#ifndef ABLATELIBRARY_CONSTANTPRESSUREFIX_HPP
#define ABLATELIBRARY_CONSTANTPRESSUREFIX_HPP

#include <memory>
#include "eos/eos.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "flowProcess.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::finiteVolume::processes {

/**
 * This classes updates density after each time step to ensure pressure remains constant
 */
class ConstantPressureFix : public FlowProcess {
   private:
    //! The constant reference pressure
    const PetscReal pressure;

    //! Store the equation of state to compute pressure
    const std::shared_ptr<eos::EOS> eos;

    //! store the function needed to compute density
    ablate::eos::ThermodynamicFunction densityFunction;

    //! store the function needed to compute sensibleInternalEnergy
    ablate::eos::ThermodynamicFunction internalSensibleEnergyFunction;

    //! function to compute euler from energy and pressure
    ablate::eos::EOSFunction eulerFromEnergyAndPressure = nullptr;

    //! function to compute densityYi from energy and pressure
    ablate::eos::EOSFunction densityOtherPropFromEnergyAndPressure = nullptr;

   public:
    explicit ConstantPressureFix(std::shared_ptr<eos::EOS> eos, double pressure);

    /**
     * Function to setup UpdateDensityForConstantPressure
     * @return
     */
    void Setup(ablate::finiteVolume::FiniteVolumeSolver& fv) override;
};

}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_CONSTANTPRESSUREFIX_HPP
