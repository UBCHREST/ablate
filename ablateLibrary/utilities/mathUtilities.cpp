#include "mathUtilities.hpp"

/**************************************************************************/
static const PetscScalar SMALL = 1e-10;

static inline void TangentialVector0(PetscInt dim, const PetscScalar *nVec, PetscScalar *tangVec) {
    //_tang_vect should be dimensioned
    switch (dim) {
        case 1:  // tangential vectors are simply j and k,should never be called?
            tangVec[0] = 0.;
            tangVec[1] = 1.;
            tangVec[2] = 0.;
            break;
        case 2:
            // tangential vectors are the solution of 2 algebraic equations
            tangVec[0] = -nVec[1];  // t1
            tangVec[1] = nVec[0];   // t2
            //_tang_vect[2] = 0.;not needed for ndims ==2, there should be no 3rd component
            break;
        case 3:
            // 3D tangential vectors...a little more complex
            auto ny = nVec[1] + SMALL, nz = nVec[2] + SMALL;
            auto one_m_nz2 = 1. - nz * nz;
            tangVec[0] = -(nz * ny * ny) / (one_m_nz2 + SMALL);
            tangVec[1] = nz * ny * nz / (one_m_nz2 + SMALL);
            tangVec[2] = 0.;  // fix this, it is arbitrary
            // now need to re-normalize these vectors
            ablate::utilities::MathUtilities::NormVector(dim, tangVec);
            break;
    }
}
/**************************************************************************/
static inline void TangentialVector1(PetscInt dim, const PetscScalar *nVec, PetscScalar *tangVec) {
    //		   _tang_vect should be dimensioned as .... new double[2][3]
    switch (dim) {
        case 1:  // tangential vectors are simply j and k, should never be called?
            tangVec[0] = 0.;
            tangVec[1] = 0.;
            tangVec[2] = 1.;
            break;
        case 2:
            // other vect is just k vect
            tangVec[0] = 0.;
            tangVec[1] = 0.;
            //_tang_vect[2] = 1.;not needed for ndims ==2, there should be no 3rd component
            break;
        case 3:
            // 3D tangential vectors...a little more complex
            auto nx = nVec[0] + SMALL, ny = nVec[1] + SMALL, nz = nVec[2] + SMALL;
            auto one_m_nz2 = 1. - nz * nz;
            tangVec[0] = nx / (ny + SMALL);
            tangVec[1] = 1.;  // fix this, it is arbitrary
            tangVec[2] = -one_m_nz2 / (ny * nz + SMALL);
            // now need to re-normalize these vectors
            ablate::utilities::MathUtilities::NormVector(dim, tangVec);
            break;
    }
}

void ablate::utilities::MathUtilities::ComputeTransformationMatrix(PetscInt dim, const PetscScalar *normal, PetscScalar tm[3][3]) {
    // Place the TM[0] using the normal
    PetscArraycpy(tm[0], normal, dim);

    // Compute the tangential vectors
    switch (dim) {
        case 3:
            TangentialVector1(dim, normal, tm[2]);
            [[fallthrough]];
        case 2:
            TangentialVector0(dim, normal, tm[1]);
    }
}
