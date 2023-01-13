#ifndef ABLATELIBRARY_BOUNDARYSOLVERMONITOR_HPP
#define ABLATELIBRARY_BOUNDARYSOLVERMONITOR_HPP

#include <petsc.h>
#include "boundarySolver/boundarySolver.hpp"
#include "domain/region.hpp"
#include "domain/subDomain.hpp"
#include "eos/eos.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::monitors {

/**
 * use to call the boundary solver to output any specific output variables
 */
class BoundarySolverMonitor : public Monitor, public io::Serializable {
   private:
    /**
     * Named used for output
     */
    std::string name = "_monitor";

    /**
     * The boundary solver
     */
    std::shared_ptr<ablate::boundarySolver::BoundarySolver> boundarySolver;

    /**
     * This dm contains all cells, faces, and nodes with a label/marker for the boundary field
     */
    DM boundaryDm = nullptr;

    /**
     * This dm contains only the faces on the boundary and an output boundary field
     */
    DM faceDm = nullptr;

   public:
    /**
     * Clean up the petsc objects
     */
    ~BoundarySolverMonitor() override;

    /**
     * Register this solver with the boundary solver
     * @param solver the solver must be of type ablate::boundarySolver::BoundarySolver
     */
    void Register(std::shared_ptr<solver::Solver> solver) override;

    /**
     * This is not needed because this is only called upon serialize.
     * @return
     */
    PetscMonitorFunction GetPetscFunction() override { return nullptr; }

    /**
     *  Should be unique for the monitor
     * @return
     */
    const std::string& GetId() const override { return name; }

    /**
     * Compute and store the current boundary output values
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override;

    /**
     * This is not needed for ghe boundary solver monitor
     * @param viewer
     * @param sequenceNumber
     * @param time
     */
    void Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) override {}
};

}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_BOUNDARYSOLVERMONITOR_HPP
