#include "evTransport.hpp"
#include <utilities/mathUtilities.hpp>
#include "eulerTransport.hpp"

PetscErrorCode ablate::finiteVolume::processes::EVTransport::UpdateNonConservedEV(PetscReal time, PetscInt dim, const PetscFVCellGeom *cellGeom, const PetscInt *uOff,
                                                                                  const PetscScalar *conservedValues, PetscScalar *auxField, void *ctx) {
    PetscFunctionBeginUser;
    PetscReal density = conservedValues[uOff[0] + EulerAdvection::RHO];
    auto evTransport = (ablate::finiteVolume::processes::EVTransport*)ctx;

    for (PetscInt ev = 0; ev < evTransport->numberEV; ev++) {
        auxField[ev] = conservedValues[uOff[1] + ev] / density;
    }

    PetscFunctionReturn(0);
}

void ablate::finiteVolume::processes::EVTransport::Initialize(ablate::finiteVolume::FiniteVolume &flow) {
    // Update the auxField
    flow.RegisterAuxFieldUpdate(UpdateNonConservedEV, this, nonConserved, {"euler", conserved});


}

void ablate::finiteVolume::processes::EVTransport::DecodeEulerState(PetscInt dim, const PetscReal* conservedValues, const PetscReal* normal, PetscReal* density, PetscReal* normalVelocity, PetscReal* velocity) {
    // decode
    *density = conservedValues[EulerAdvection::RHO];

    // Get the velocity in this direction
    (*normalVelocity) = 0.0;
    for (PetscInt d = 0; d < dim; d++) {
        velocity[d] = conservedValues[EulerAdvection::RHOU + d] / (*density);
        (*normalVelocity) += velocity[d] * normal[d];
    }
}


PetscErrorCode ablate::finiteVolume::processes::EVTransport::EVFlux(PetscInt dim, const PetscFVFaceGeom *fg, const PetscInt *uOff, const PetscInt *uOff_x, const PetscScalar *fieldL,
                                                                 const PetscScalar *fieldR, const PetscScalar *gradL, const PetscScalar *gradR, const PetscInt *aOff, const PetscInt *aOff_x,
                                                                 const PetscScalar *auxL, const PetscScalar *auxR, const PetscScalar *gradAuxL, const PetscScalar *gradAuxR, PetscScalar *fL,
                                                                 void *ctx) {
    PetscFunctionBeginUser;
    auto evTransport = (ablate::finiteVolume::processes::EVTransport*)ctx;

    // this order is based upon the order that they are passed into RegisterRHSFunction
    const int EULER_FIELD_ID = 0;
    const int DENSITY_EV_FIELD_ID = 1;
    const int EV_FIELD_ID = 0;

    // Compute the norm
    PetscReal norm[3];
    ablate::utilities::MathUtilities::NormVector(dim, fg->normal, norm);
    const PetscReal areaMag = ablate::utilities::MathUtilities::MagVector(dim, fg->normal);

    const int EULER_FIELD = 0;
    const int YI_FIELD = 1;

    // Decode the left and right states
    PetscReal densityL;
    PetscReal normalVelocityL;
    PetscReal velocityL[3];
    DecodeEulerState(dim, fieldL + uOff[EULER_FIELD_ID], norm, &densityL, &normalVelocityL, velocityL);

    PetscReal densityR;
    PetscReal normalVelocityR;
    PetscReal velocityR[3];
    DecodeEulerState(dim, fieldR + uOff[EULER_FIELD_ID], norm, &densityR, &normalVelocityR, velocityR);

    // get the face values
    PetscReal massFlux;


    if (evTransport->fluxCalculatorFunction(evTransport->fluxCalculatorCtx, normalVelocityL, aL, densityL, pL, normalVelocityR, aR, densityR, pR, &massFlux, NULL) ==
        fluxCalculator::LEFT) {
        // march over each gas species
        for (PetscInt sp = 0; sp < eulerAdvectionData->numberSpecies; sp++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[sp] = (massFlux * fieldL[uOff[YI_FIELD] + sp] / densityL) * areaMag;
        }
    } else {
        // march over each gas species
        for (PetscInt sp = 0; sp < eulerAdvectionData->numberSpecies; sp++) {
            // Note: there is no density in the flux because uR and UL are density*yi
            flux[sp] = (massFlux * fieldR[uOff[YI_FIELD] + sp] / densityR) * areaMag;
        }
    }

    PetscFunctionReturn(0);


    PetscFunctionReturn(0);

}
