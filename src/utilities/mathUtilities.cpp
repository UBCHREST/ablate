#include "mathUtilities.hpp"

/**************************************************************************/
void ablate::utilities::MathUtilities::ComputeTransformationMatrix(PetscInt dim, const PetscScalar* normal, PetscScalar tm[3][3]) {
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

PetscErrorCode ablate::utilities::MathUtilities::ComputeNorm(ablate::utilities::MathUtilities::Norm normType, Vec x, Vec y, PetscReal* norm) {
    PetscFunctionBeginUser;
    // Compute the difference between x and y and put it in y
    PetscCall(VecAXPY(y, -1.0, x));

    // determine the type of norm to ask of petsc
    NormType petscNormType;
    switch (normType) {
        case Norm::L1_NORM:
        case Norm::L1:
            petscNormType = NORM_1;
            break;
        case Norm::L2_NORM:
        case Norm::L2:
            petscNormType = NORM_2;
            break;
        case Norm::LINF:
            petscNormType = NORM_INFINITY;
            break;
        default:
            std::stringstream error;
            error << "Unable to process norm type " << normType;
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "%s", error.str().c_str());
    }

    // compute the norm along the stride
    PetscCall(VecStrideNormAll(y, petscNormType, norm));

    // Get the total size of norm from the block size
    PetscInt totalComponents;
    PetscCall(VecGetBlockSize(y, &totalComponents));

    // normalize the error if _norm
    if (normType == Norm::L1_NORM) {
        PetscInt size;
        VecGetSize(y, &size);
        PetscReal factor = (1.0 / ((PetscReal)size / totalComponents));
        for (PetscInt c = 0; c < totalComponents; c++) {
            norm[c] *= factor;
        }
    }
    if (normType == Norm::L2_NORM) {
        PetscInt size;
        VecGetSize(y, &size);
        PetscReal factor = PetscSqrtReal(1.0 / ((PetscReal)size / totalComponents));
        for (PetscInt c = 0; c < totalComponents; c++) {
            norm[c] *= factor;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

std::ostream& ablate::utilities::operator<<(std::ostream& os, const ablate::utilities::MathUtilities::Norm& v) {
    switch (v) {
        case MathUtilities::Norm::L1:
            return os << "l1";
        case MathUtilities::Norm::L1_NORM:
            return os << "l1_norm";
        case MathUtilities::Norm::L2:
            return os << "l2";
        case MathUtilities::Norm::LINF:
            return os << "linf";
        case MathUtilities::Norm::L2_NORM:
            return os << "l2_norm";
        default:
            return os;
    }
}

std::istream& ablate::utilities::operator>>(std::istream& is, ablate::utilities::MathUtilities::Norm& v) {
    std::string enumString;
    is >> enumString;

    if (enumString == "l2") {
        v = MathUtilities::Norm::L2;
    } else if (enumString == "linf") {
        v = MathUtilities::Norm::LINF;
    } else if (enumString == "l2_norm") {
        v = MathUtilities::Norm::L2_NORM;
    } else if (enumString == "l1_norm") {
        v = MathUtilities::Norm::L1_NORM;
    } else if (enumString == "l1") {
        v = MathUtilities::Norm::L1;
    } else {
        throw std::invalid_argument("Unknown norm type " + enumString);
    }
    return is;
}
