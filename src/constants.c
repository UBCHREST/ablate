#include "constants.h"

void PackFlowParameters(FlowParameters *parameters, PetscScalar *constantArray ){
constantArray[STROUHAL] = parameters->strouhal;
constantArray[REYNOLDS] = parameters->reynolds;
constantArray[FROUDE] = parameters->froude;
constantArray[PECLET] = parameters->peclet;
constantArray[HEATRELEASE] = parameters->heatRelease;
constantArray[GAMMA] = parameters->gamma;
constantArray[MU] = parameters->mu;
constantArray[K] = parameters->k;
constantArray[CP] = parameters->cp;
constantArray[BETA] = parameters->beta;
}