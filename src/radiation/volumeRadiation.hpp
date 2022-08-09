#ifndef ABLATELIBRARY_VOLUMERADIATION_HPP
#define ABLATELIBRARY_VOLUMERADIATION_HPP

#include "radiation.hpp"
namespace ablate::radiation {

class VolumeRadiation : public Radiation, public solver::CellSolver {
    /**
     * Function passed into PETSc to compute the FV RHS
     * @param dm
     * @param time
     * @param locXVec
     * @param globFVec
     *
     * @param ctx
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec);

    void Initialize() override;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_VOLUMERADIATION_HPP
