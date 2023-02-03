#ifndef ABLATELIBRARY_VIRTUALTCP_H
#define ABLATELIBRARY_VIRTUALTCP_H

#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "io/interval/fixedInterval.hpp"
#include "radiation/radiation.hpp"

namespace ablate::monitors {

/**
 * use to call the boundary solver to output any specific output variables
 */
class VirtualTcp : public Monitor, public io::Serializable {

   public:
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
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_VIRTUALTCP_H
