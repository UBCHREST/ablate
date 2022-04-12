#include "mathUtilities.hpp"

/**************************************************************************/
void ablate::utilities::MathUtilities::ComputeTransformationMatrix(PetscInt dim, const PetscScalar *normal, PetscScalar tm[3][3]) {
    // Place the TM[0] using the normal
    PetscArraycpy(tm[0], normal, dim);

    // Compute the tangential vectors
    switch (dim) {
        case 3: {
            PetscScalar refAxis[3] = {0.0, 0.0, 1.0};
            if (PetscAbs(normal[2]) > .9) {
                refAxis[1] = 1.0;
                refAxis[2] = 0.0;
            }
            // take normal x ref axis to compute t1
            tm[1][0] = normal[1] * refAxis[2] - normal[2] * refAxis[1];
            tm[1][1] = normal[2] * refAxis[0] - normal[0] * refAxis[2];
            tm[1][2] = normal[0] * refAxis[1] - normal[1] * refAxis[0];
            NormVector(dim, tm[1]);

            // take the normal x t1 axis
            tm[2][0] = normal[1] * tm[1][2] - normal[2] * tm[1][1];
            tm[2][1] = normal[2] * tm[1][0] - normal[0] * tm[1][2];
            tm[2][2] = normal[0] * tm[1][1] - normal[1] * tm[1][0];

        } break;
        case 2:
            // tangential vectors are the solution of 2 algebraic equations
            tm[1][0] = -normal[1];  // t1
            tm[1][1] = normal[0];   // t2
    }
}

PetscReal ablate::utilities::MathUtilities::ComputeDeterminant(PetscInt dim, PetscScalar (*A)[3]) {
    switch (dim) {
        case 1:
            return A[0][0];
        case 2:
            return A[0][0] * A[1][1] - A[0][1] * A[1][0];
        case 3:
            return A[0][0] * (A[2][2] * A[1][1] - A[2][1] * A[1][2]) - A[1][0] * (A[2][2] * A[0][1] - A[2][1] * A[0][2]) + A[2][0] * (A[1][2] * A[0][1] - A[1][1] * A[0][2]);
        default:
            throw std::invalid_argument("The ablate::utilities::MathUtilities::ComputeDeterminant must be size 3 or less.");
    }
}
