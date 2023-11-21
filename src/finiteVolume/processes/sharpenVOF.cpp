#include "sharpenVOF.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
ablate::finiteVolume::processes::SharpenVOF::SharpenVOF(std::vector<double> tol) : tol(tol) {}

void ablate::finiteVolume::processes::SharpenVOF::Setup(ablate::finiteVolume::FiniteVolumeSolver &fv) {
    // add the source function

    fv.RegisterPreRHSFunction(SharpenInterface, this);
}


PetscErrorCode SharpenInterface(ablate::finiteVolume::FiniteVolumeSolver *fv, TS ts, PetscReal time, bool initialStage, Vec locX, void *ctx) {
    PetscFunctionBeginUser;

    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::finiteVolume::processes::Process, ablate::finiteVolume::processes::SharpenVOF, "sharpen a VOF field before computing the RHS.",
         ARG(std::vector<double>, "tol", "tolerances to use"));
