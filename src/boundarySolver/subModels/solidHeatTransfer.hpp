#ifndef ABLATELIBRARY_SOLIDHEATTRANSFER_HPP
#define ABLATELIBRARY_SOLIDHEATTRANSFER_HPP

#include <memory>
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::boundarySolver::subModels {

class SolidHeatTransfer {
   private:
    // Hold the orginal DM, this DM is shared by both TSs
    DM subModelDM;
    // Hold a TS for heating
    TS heatingTS;

    // Hold a separate TS for sublimation
    TS sublimationTS;

   public:
    /**
     * Create a single 1D solid model
     * @param parameters
     */
    SolidHeatTransfer(std::shared_ptr<ablate::parameters::Parameters> parameters);

    /**
     * Clean up the petsc objects
     */
    ~SolidHeatTransfer();
};

}  // namespace ablate::boundarySolver::subModels
#endif  // ABLATELIBRARY_SOLIDHEATTRANSFER_HPP
