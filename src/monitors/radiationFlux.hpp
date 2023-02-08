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
class RadiationFlux : public Monitor, public io::Serializable {
   private:
    /**
     * This is the DM that the monitor uses to place ray tracing particles on the faces.
     */
    DM monitorDm = nullptr;
    /**
     * This dm contains only the faces on the boundary and an output boundary field
     */
    DM fluxDm = nullptr;

    /**
     * Named used for output
     */
    std::string name = "_radiationFluxMonitor";

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
    solver::DynamicRange monitorRange;

    /**
     * Region for the radiation solver to monitor
     */
    std::shared_ptr<ablate::domain::Region> radiationFluxRegion;

   public:
    RadiationFlux(std::vector<std::shared_ptr<radiation::Radiation>> radiationIn, std::shared_ptr<domain::Region> radiationFluxRegionIn);

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
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * return context to be returned to the PetscMonitorFunction.  By default this is a pointer to this instance
     */
    void* GetContext() override { return this; }

    [[nodiscard]] const std::string& GetId() const override { return name; };

    void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override {};

    [[nodiscard]] bool Serialize() const override { return true; }

    /**
     * This is not needed because this is only called upon serialize.
     * @return
     */
    PetscMonitorFunction GetPetscFunction() override { return nullptr; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_VIRTUALTCP_H
