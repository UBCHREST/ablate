#ifndef ABLATELIBRARY_LODIBOUNDARY_HPP
#define ABLATELIBRARY_LODIBOUNDARY_HPP

#include "boundarySolver/boundaryProcess.hpp"
#include "eos/eos.hpp"

namespace ablate::boundarySolver::lodi{
class LODIBoundary : public BoundaryProcess{
   protected:
    const std::shared_ptr<eos::EOS> eos;
   public:
    explicit LODIBoundary(std::shared_ptr<eos::EOS> eos);

    void Initialize(ablate::boundarySolver::BoundarySolver& bSolver) override = 0;



};


}
#endif  // ABLATELIBRARY_LODIBOUNDARY_HPP
