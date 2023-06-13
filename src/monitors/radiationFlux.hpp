#ifndef ABLATELIBRARY_VIRTUALTCP_H
#define ABLATELIBRARY_VIRTUALTCP_H

#include "boundarySolver/boundarySolver.hpp"
#include "domain/dynamicRange.hpp"
#include "io/interval/fixedInterval.hpp"
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "radiation/radiation.hpp"
#include "radiation/surfaceRadiation.hpp"

namespace ablate::monitors {

/**
 * use to call the boundary solver to output any specific output variables
 */
class RadiationFlux : public Monitor, public io::Serializable {
   private:
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
    std::vector<std::shared_ptr<ablate::radiation::SurfaceRadiation>> radiation;

    /**
     * The boundary solver
     */
    std::shared_ptr<ablate::boundarySolver::BoundarySolver> boundarySolver;

    /**
     * Face range stored by the radiation to locate the
     */
    ablate::domain::DynamicRange monitorRange;

    /**
     * Region for the radiation solver to monitor
     */
    std::shared_ptr<ablate::domain::Region> radiationFluxRegion;

   public:
    RadiationFlux(std::vector<std::shared_ptr<radiation::SurfaceRadiation>> radiationIn, std::shared_ptr<domain::Region> radiationFluxRegionIn,
                  std::shared_ptr<ablate::monitors::logs::Log> = {});

    /**
     * Clean up the petsc objects
     */
    ~RadiationFlux() override;

    /**
     * Register this solverIn with the boundary solverIn
     * @param solverIn which contains the region being monitored.
     */
    void Register(std::shared_ptr<solver::Solver> solverIn) override;

    /**
     * Compute and store the current boundary output values
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    PetscErrorCode Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * return context to be returned to the PetscMonitorFunction.  By default this is a pointer to this instance
     */
    void* GetContext() override { return this; }

    [[nodiscard]] const std::string& GetId() const override { return name; };

    PetscErrorCode Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override { return 0; };

    [[nodiscard]] bool Serialize() const override { return true; }

    /**
     * This is not needed because this is only called upon serialize.
     * @return
     */
    PetscMonitorFunction GetPetscFunction() override { return nullptr; }

   protected:
    const std::shared_ptr<ablate::monitors::logs::Log> log = nullptr; //! Monitor output, primarily for testing purposes.

};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_VIRTUALTCP_H
