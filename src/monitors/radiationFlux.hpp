#ifndef ABLATELIBRARY_VIRTUALTCP_H
#define ABLATELIBRARY_VIRTUALTCP_H

#include "boundarySolver/boundarySolver.hpp"
#include "io/interval/fixedInterval.hpp"
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "radiation/radiation.hpp"
#include "solver/dynamicRange.hpp"

namespace ablate::monitors {

/**
 * use to call the boundary solver to output any specific output variables
 */
class RadiationFlux : public Monitor {

   private:
    /**
     * A vector of radiation models which describe ray tracers with different attached properties.
     */
    std::vector<std::shared_ptr<ablate::radiation::Radiation>> radiation;

    /**
     * The boundary solver
     */
    std::shared_ptr<ablate::boundarySolver::BoundarySolver> boundarySolver;

    /**
     * Face range stored by the radiation to locate the
     */
    solver::DynamicRange faceRange;

    /**
     * Region for the radiation solver to monitor
     */
    std::shared_ptr<ablate::domain::Region> radiationRegion;

   public:
    RadiationFlux(std::vector<std::shared_ptr<radiation::Radiation>> radiationIn, std::shared_ptr<domain::Region> radiationRegion);

    /**
     * Clean up the petsc objects
     */
    ~RadiationFlux() override;

    /**
     * Register this solver with the boundary solver
     * @param solver which contains the region being monitored.
     */
    void Register(std::shared_ptr<solver::Solver> solver) override;

    /**
     * Compute and store the current boundary output values
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    static PetscErrorCode MonitorRadiation(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);

    /**
     * return context to be returned to the PetscMonitorFunction.  By default this is a pointer to this instance
     */
    void* GetContext() override { return this; }

    /**
     * This is not needed because this is only called upon serialize.
     * @return
     */
    PetscMonitorFunction GetPetscFunction() override { return MonitorRadiation; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_VIRTUALTCP_H
