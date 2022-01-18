#ifndef ABLATELIBRARY_PRESSUREGRADIENTSCALING_HPP
#define ABLATELIBRARY_PRESSUREGRADIENTSCALING_HPP

#include <memory>
#include "eos/eos.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "flowProcess.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::finiteVolume::processes {
/**
 * Rescales the thermodynamic pressure gradient scaling the acoustic propagation speeds to allow for a larger time step.
 * See:
 *   DesJardin, Paul E., Timothy J. O’Hern, and Sheldon R. Tieszen. "Large eddy simulation and experimental measurements of the near-field of a large turbulent helium plume." Physics of fluids 16.6
 * (2004): 1866-1883.
 */
class PressureGradientScaling : public FlowProcess {
   private:
    /**
     * Store the equation of state to compute pressure
     */
    const std::shared_ptr<eos::EOS> eos;

    /**
     * The maximum allowed mach number
     */
    const PetscReal maxMachAllowed = 0.7;

    /**
     * avoid alpha jumping up to quickly if maxdelPfac=0
     */
    const PetscReal maxAlphaChange = 0.10;

    /**
     * The maximum allowed alpha
     */
    const PetscReal maxAlphaAllowed;

    /**
     * max variation from mean pressure
     */
    const PetscReal maxDeltaPressureFac;

    /**
     * The reference length of the domain
     */
    const PetscReal domainLength;

    /**
     * Store a log used to output the required information
     */
    const std::shared_ptr<ablate::monitors::logs::Log> log;

    /**
     * Store current updated components
     */
    PetscReal maxMach = 0.0;
    PetscReal alpha;

   public:
    PressureGradientScaling(std::shared_ptr<eos::EOS> eos, double alphaInit, double domainLength, double maxAlphaAllowed = {}, double maxDeltaPressureFac = {},
                            std::shared_ptr<ablate::monitors::logs::Log> = {});

    /**
     * function to compute the average density in the domain
     * @param flowTs
     * @param flow
     * @return
     */
    PetscErrorCode UpdatePreconditioner(TS flowTs, ablate::solver::Solver& flow);

    /**
     * Function to setup timestepping with the PressureGradientScaling.  This can be called multiple times and will only be registered once
     * @return
     */
    void Initialize(ablate::finiteVolume::FiniteVolumeSolver& fv) override;

    // Alpha accessor
    inline const PetscReal& GetAlpha() { return alpha; }

    // Alpha accessor
    inline const PetscReal& GetMaxMach() { return maxMach; }
};
}  // namespace ablate::finiteVolume::processes
#endif  // ABLATELIBRARY_PRESSUREGRADIENTSCALING_HPP
