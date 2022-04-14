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

   public:
    explicit ConstantPressureFix(std::shared_ptr<eos::EOS> eos, double pressure);

    /**
     * Function to setup UpdateDensityForConstantPressure
     * @return
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& fv) override;

};

}
#endif  // ABLATELIBRARY_CONSTANTPRESSUREFIX_HPP
