#ifndef ABLATELIBRARY_MATHUTILITIES_HPP
#define ABLATELIBRARY_MATHUTILITIES_HPP
#include <petsc.h>

namespace ablate::utilities {
class MathUtilities {
   public:
    template <class I, class T>
    static inline void ScaleVector(I dim, T* vec, T alpha) {
        for (I d = 0; d < dim; d++) {
            vec[d] *= alpha;
        }
    }

    template <class I, class T>
    static inline void NormVector(I dim, T* vec) {
        auto mag = MagVector(dim, vec);
        for (I d = 0; d < dim; d++) {
            vec[d] = vec[d] / mag;
        }
    }

    template <class I, class T>
    static inline void NormVector(I dim, const T* in, T* out) {
        T mag = 0.0;
        for (I d = 0; d < dim; d++) {
            mag += in[d] * in[d];
        }
        mag = PetscSqrtReal(mag);
        for (I d = 0; d < dim; d++) {
            out[d] = in[d] / mag;
        }
    }

    template <class I, class T>
    static inline T MagVector(I dim, const T* in) {
        T mag = 0.0;
        for (I d = 0; d < dim; d++) {
            mag += in[d] * in[d];
        }
        return PetscSqrtReal(mag);
    }

    template <class I, class T>
    static inline T DotVector(I dim, const T* a, const T* b) {
        T dot = 0.0;
        for (I d = 0; d < dim; d++) {
            dot += a[d] * b[d];
        }
        return dot;
    }

    /**
     * subtract so that c = a - b
     * @tparam I
     * @tparam T
     * @param dim
     * @param a
     * @param b
     */
    template <class I, class T>
    static inline void Subtract(I dim, const T* a, const T* b, T* c) {
        for (I i = 0; i < dim; i++) {
            c[i] = a[i] - b[i];
        }
    }

    /**
     *  matrix vector multiplication
     *  returns out = [A]*in
     *  1st index in A is row, 2nd is col
     * @param dim
     * @param transformationMatrix
     */
    template <class I, class T>
    static inline void Multiply(I dim, const T A[3][3], const T* in, T* out) {
        for (I i = 0; i < dim; i++) {
            out[i] = 0.0;
            for (I j = 0; j < dim; j++) {
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
    template <class I, class T>
    static inline void MultiplyTranspose(I dim, const T A[3][3], const T* in, T* out) {
        for (I i = 0; i < dim; i++) {
            out[i] = 0.0;
            for (I j = 0; j < dim; j++) {
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
