#ifndef ABLATELIBRARY_DEBUGBOUNDARYSOLVER_HPP
#define ABLATELIBRARY_DEBUGBOUNDARYSOLVER_HPP

#include "boundarySolver.hpp"

namespace ablate::boundarySolver {

/**
 * this is a debug extension of the boundary solver that outputs useful information for debugging problems with the boundary solver
 */
class DebugBoundarySolver : public BoundarySolver {
   public:
    /**
     *
     * @param solverId the id for this solver
     * @param region the boundary cell region
     * @param fieldBoundary the region describing the faces between the boundary and field
     * @param boundaryProcesses a list of boundary processes
     * @param options other options
     */
    DebugBoundarySolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary, std::vector<std::shared_ptr<BoundaryProcess>> boundaryProcesses,
                        std::shared_ptr<parameters::Parameters> options, bool mergeFaces = false);

    /**
     * override setup to allow outputting of the stencil per rank
     */
    void Setup() override;

    /**
     * Override the ComputeRHSFunction to output the locFVec at every boundary point
     * @param dm
     * @param time
     * @param locXVec
     * @param globFVec
     * @param ctx
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) override;

   private:
    /**
     * helper function to output the cell location
     * @param stream
     * @param cell
     */
    void OutputStencilCellLocation(std::ostream& stream, PetscInt cell);
};

}  // namespace ablate::boundarySolver
#endif  // ABLATELIBRARY_DEBUGBOUNDARYSOLVER_HPP
