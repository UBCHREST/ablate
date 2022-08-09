#include "volumeRadiation.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"

void ablate::radiation::VolumeRadiation::Initialize() {
    solver::Range cellRange;
    GetCellRange(cellRange);
    ablate::radiation::Radiation::Initialize(cellRange);
    RestoreRange(cellRange);
}

PetscErrorCode ablate::radiation::VolumeRadiation::ComputeRHSFunction(PetscReal time, Vec solVec, Vec rhs) {
    PetscFunctionBegin;

    /** Get the array of the local f vector, put the intensity into part of that array instead of using the radiative gain variable. */
    const PetscScalar* rhsArray;
    VecGetArrayRead(rhs, &rhsArray);

    ablate::radiation::Radiation::Solve(solVec, rhs);

    solver::Range cellRange;
    GetCellRange(cellRange);

    for (PetscInt c = cellRange.start; c < cellRange.end; ++c) {            //!< This will iterate only though local cells
        const PetscInt iCell = cellRange.points ? cellRange.points[c] : c;  //!< Isolates the valid cells
        PetscScalar* rhsValues;
        rhsValues[ablate::finiteVolume::CompressibleFlowFields::RHOE] += origin[iCell].intensity;  // GetIntensity(iCell);  //!< Loop through the cells and update the equation of state
    }
    PetscFunctionReturn(0);
}