#ifndef ABLATELIBRARY_BOUNDARYPROCESS_HPP
#define ABLATELIBRARY_BOUNDARYPROCESS_HPP

#include "boundarySolver.hpp"

namespace ablate::boundarySolver {

class BoundaryProcess {
   public:
    virtual ~BoundaryProcess() = default;
    virtual void Initialize(ablate::boundarySolver::BoundarySolver& bSolver) = 0;

};

}
#endif  // ABLATELIBRARY_BOUNDARYPROCESS_HPP
