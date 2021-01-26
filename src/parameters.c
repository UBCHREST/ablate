#include "parameters.h"

void PackFlowParameters(FlowParameters *parameters, PetscScalar *constantArray) {
    constantArray[STROUHAL] = parameters->strouhal;
    constantArray[REYNOLDS] = parameters->reynolds;
    constantArray[FROUDE] = parameters->froude;
    constantArray[PECLET] = parameters->peclet;
    constantArray[HEATRELEASE] = parameters->heatRelease;
    constantArray[PTH] = parameters->pth;
    constantArray[GAMMA] = parameters->gamma;
    constantArray[MU] = parameters->mu;
    constantArray[K] = parameters->k;
    constantArray[CP] = parameters->cp;
    constantArray[BETA] = parameters->beta;
    constantArray[GRAVITY_DIRECTION] = parameters->gravityDirection;
}

PetscErrorCode SetupFlowParameters(PetscBag *flowParametersBag){
    FlowParameters *p;
    PetscErrorCode ierr;

    PetscFunctionBeginUser;
    // create an empty bag
    PetscBagCreate(PETSC_COMM_WORLD,sizeof(FlowParameters),flowParametersBag);

    // setup PETSc parameter bag
    ierr = PetscBagGetData(*flowParametersBag, (void **)&p);CHKERRQ(ierr);
    ierr = PetscBagSetName(*flowParametersBag, "flowParameters", "Low Mach Flow Parameters");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->strouhal, 1.0, "strouhal", "Strouhal number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->reynolds, 1.0, "reynolds", "Reynolds number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->froude, 1.0, "froude", "Feynolds number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->peclet, 1.0, "peclet", "Peclet number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->heatRelease, 1.0, "heatRelease", "Heat Release number");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->gamma, 1.0, "gamma", "gamma: p_o/(\rho_o Cp_o T_o");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->pth, 1.0, "pth", "non-dimensional thermodyamic pressure");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->mu, 1.0, "mu", "non-dimensional viscosity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->k, 1.0, "k", "non-dimensional thermal conductivity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->cp, 1.0, "cp", "non-dimensional specific heat capacity");CHKERRQ(ierr);
    ierr = PetscBagRegisterReal(*flowParametersBag, &p->beta, 1.0, "beta", "non-dimensional thermal expansion coefficient ");CHKERRQ(ierr);
    ierr = PetscBagRegisterInt(*flowParametersBag, &p->gravityDirection, 1, "gravityDirection", "axis");CHKERRQ(ierr);

    PetscFunctionReturn(0);
}
