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
class VirtualTcp : public Monitor, public io::Serializable {

   private:
    /**
     * A vector of radiation models which describe ray tracers with different attached properties.
     */
    std::vector<ablate::radiation::Radiation> radiation;

    /**
     * The boundary solver
     */
    std::shared_ptr<ablate::boundarySolver::BoundarySolver> boundarySolver;

    /**
     * Face range stored by the radiation to locate the
     */
    solver::DynamicRange faceRange;

   public:

    VirtualTcp(std::vector<radiation::Radiation> radiationIn);

    /**
     * Clean up the petsc objects
     */
    ~VirtualTcp() override;

    /**
     * Register this solver with the boundary solver
     * @param solver the solver must be of type ablate::boundarySolver::BoundarySolver
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
     * This is not needed because this is only called upon serialize.
     * @return
     */
    PetscMonitorFunction GetPetscFunction() override { return nullptr; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_VIRTUALTCP_H
