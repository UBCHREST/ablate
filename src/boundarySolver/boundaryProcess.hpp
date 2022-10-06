#ifndef ABLATELIBRARY_BOUNDARYPROCESS_HPP
#define ABLATELIBRARY_BOUNDARYPROCESS_HPP

#include "boundarySolver.hpp"

namespace ablate::boundarySolver {

class BoundaryProcess {
   public:
    virtual ~BoundaryProcess() = default;
    /**
     * Setup up all functions not dependent upon the mesh
     * @param fv
     */
    virtual void Setup(ablate::boundarySolver::BoundarySolver& fv) = 0;
    /**
     * Set up mesh dependent initialization
     * @param fv
     */
    virtual void Initialize(ablate::boundarySolver::BoundarySolver& fv){};
};

}  // namespace ablate::boundarySolver
#endif  // ABLATELIBRARY_BOUNDARYPROCESS_HPP
