#ifndef ABLATELIBRARY_MATHUTILITIES_HPP
#define ABLATELIBRARY_MATHUTILITIES_HPP
#include <petsc.h>

namespace ablate::utilities {
class MathUtilities {
   public:
    static inline void NormVector(PetscInt dim, PetscReal* vec) {
        auto mag = MagVector(dim, vec);
        for (PetscInt d = 0; d < dim; d++) {
            vec[d] = vec[d] / mag;
        }
    }

    static inline void NormVector(PetscInt dim, const PetscReal* in, PetscReal* out) {
        PetscReal mag = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            mag += in[d] * in[d];
        }
        mag = PetscSqrtReal(mag);
        for (PetscInt d = 0; d < dim; d++) {
            out[d] = in[d] / mag;
        }
    }

    static inline PetscReal MagVector(PetscInt dim, const PetscReal* in) {
        PetscReal mag = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            mag += in[d] * in[d];
        }
        return PetscSqrtReal(mag);
    }

    static inline PetscReal DotVector(PetscInt dim, const PetscReal* a, const PetscReal* b) {
        PetscReal dot = 0.0;
        for (PetscInt d = 0; d < dim; d++) {
            dot += a[d] * b[d];
        }
        return dot;
    }

    /**
     *  matrix vector multiplication
     *  returns out = [A]*in
     *  1st index in A is row, 2nd is col
     * @param dim
     * @param transformationMatrix
     */
    static inline void Multiply(PetscReal dim, const PetscScalar A[3][3], const PetscReal* in, PetscReal* out) {
        for (PetscInt i = 0; i < dim; i++) {
            out[i] = 0.0;
            for (PetscInt j = 0; j < dim; j++) {
                out[i] += A[i][j] * in[j];
            }
        }
    }

    /**
     *  matrix transpose vector multiplication
     *  returns out = [A]^T *in
     *  1st index in A is row, 2nd is col
     * @param dim
     * @param transformationMatrix
     */
    static inline void MultiplyTranspose(PetscReal dim, const PetscScalar A[3][3], const PetscReal* in, PetscReal* out) {
        for (PetscInt i = 0; i < dim; i++) {
            out[i] = 0.0;
            for (PetscInt j = 0; j < dim; j++) {
                out[i] += A[j][i] * in[j];
            }
        }
    }

    /**
     * Computes the transformation matrix from the global Cartesian system to a normal coordinate system.  The first component of the transformed system
     * is always the normal direction
     * @param dim
     * @param normal
     * @param transformationMatrix [row, col]
     */
    static void ComputeTransformationMatrix(PetscInt dim, const PetscScalar normal[3], PetscScalar transformationMatrix[3][3]);

    /**
     * Computes the determinant of square matrix up to size 3
     * is always the normal direction
     * @param dim
     * @param normal
     * @param transformationMatrix [row, col]
     */
    static PetscReal ComputeDeterminant(PetscInt dim, PetscScalar transformationMatrix[3][3]);

   private:
    MathUtilities() = delete;
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MATHUTILITIES_HPP
