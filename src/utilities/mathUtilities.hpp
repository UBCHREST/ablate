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
    static inline bool VectorEquals(I dim, const T* test, const T* equal, T tolerance = 1.0E-8) {
        for (I d = 0; d < dim; d++) {
            if (test[d] < equal[d] - tolerance || test[d] > equal[d] + tolerance) {
                return false;
            }
        }
        return true;
    }

    template <class R, class T>
    static inline bool Equals(R test, T equal, T tolerance = 1.0E-8) {
        return test > (equal - tolerance) && test < (equal + tolerance);
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

    template <int dim, class T>
    static inline T DotVector(const T* a, const T* b) {
        if constexpr (dim == 3) {
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        }
        if constexpr (dim == 2) {
            return a[0] * b[0] + a[1] * b[1];
        }
        if constexpr (dim == 1) {
            return a[0] * b[0];
        }
        static_assert(dim == 1 || dim == 2 || dim == 3, "ablate::utilities::MathUtilities::DotVector only support dimensions of 1, 2, or 3");
    }

    /**
     * Compute the vector and take the resulting dot product
     * result = (aE - aS) * b
     *
     * @tparam I
     * @tparam T
     * @param dim
     * @param a
     * @param b
     * @return
     */
    template <class I, class T>
    static inline T DiffDotVector(I dim, const T* aE, const T* aS, const T* b) {
        switch (dim) {  // Take the difference of the first two points and dot it with the third vector
            case 3: {
                return (((aE[0] - aS[0]) * b[0]) + ((aE[1] - aS[1]) * b[1]) + ((aE[2] - aS[2]) * b[2]));
            }
            case 2: {
                return (((aE[0] - aS[0]) * b[0]) + ((aE[1] - aS[1]) * b[1]));
            }
            default: {
                return (((aE[0] - aS[0]) * b[0]));
            }
        }
    }

    /**
     * Compute the vector and take the resulting dot product
     * result = (aE - aS) * b
     *
     * @tparam I
     * @tparam T
     * @param dim
     * @param a
     * @param b
     * @return
     */
    template <int dim, class T>
    static inline T DiffDotVector(const T* aE, const T* aS, const T* b) {
        if constexpr (dim == 3) {
            return (((aE[0] - aS[0]) * b[0]) + ((aE[1] - aS[1]) * b[1]) + ((aE[2] - aS[2]) * b[2]));
        }
        if constexpr (dim == 2) {
            return (((aE[0] - aS[0]) * b[0]) + ((aE[1] - aS[1]) * b[1]));
        }
        if constexpr (dim == 1) {
            return (((aE[0] - aS[0]) * b[0]));
        }
        static_assert(dim == 1 || dim == 2 || dim == 3, "ablate::utilities::MathUtilities::DiffDotVector only support dimensions of 1, 2, or 3");
    }

    /**
     * Take the cross of c = a x b
     * @tparam I
     * @tparam T
     * @param dim
     * @param a
     * @param b
     * @param c
     */
    template <class I, class T>
    static inline void CrossVector(I dim, const T* a, const T* b, T* c) {
        switch (dim) {
            case 3: {
                c[0] = (a[1] * b[2] - b[1] * a[2]);
                c[1] = (b[0] * a[2] - a[0] * b[2]);
                c[2] = (a[0] * b[1] - b[0] * a[1]);
                break;
            }
            case 2: {
                c[0] = (a[0] * b[1] - b[0] * a[1]);
                break;
            }
            default: {
                c[0] = 0.;
                break;
            }
        }
    }

    /**
     * Take the cross of c = a x b for a fixed size
     * @tparam I
     * @tparam T
     * @param dim
     * @param a
     * @param b
     * @param c
     */
    template <int dim, class T>
    static inline void CrossVector([[maybe_unused]] const T* a, [[maybe_unused]] const T* b, T* c) {
        if constexpr (dim == 3) {
            c[0] = (a[1] * b[2] - b[1] * a[2]);
            c[1] = (b[0] * a[2] - a[0] * b[2]);
            c[2] = (a[0] * b[1] - b[0] * a[1]);
        }
        if constexpr (dim == 2) {
            c[0] = (a[0] * b[1] - b[0] * a[1]);
        }
        if constexpr (dim == 1) {
            c[0] = 0.;
        }
        static_assert(dim == 1 || dim == 2 || dim == 3, "ablate::utilities::MathUtilities::CrossVector only support dimensions of 1, 2, or 3");
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
     * adds a to b; b = a+b
     * @tparam I
     * @tparam T
     * @param dim
     * @param a
     * @param b
     */
    template <class I, class T>
    static inline void Plus(I dim, const T* a, T* b) {
        for (I i = 0; i < dim; i++) {
            b[i] += a[i];
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
    static inline void Multiply(I dim, const T a[3][3], const T* in, T* out) {
        for (I i = 0; i < dim; i++) {
            out[i] = 0.0;
            for (I j = 0; j < dim; j++) {
                out[i] += a[i][j] * in[j];
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
    static inline void MultiplyTranspose(I dim, const T a[3][3], const T* in, T* out) {
        for (I i = 0; i < dim; i++) {
            out[i] = 0.0;
            for (I j = 0; j < dim; j++) {
                out[i] += a[j][i] * in[j];
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

    /**
     * Enum types for the compute norm function
     */
    enum class Norm { L1, L1_NORM, L2, LINF, L2_NORM };

    /**
     * Computes the specified norm between x and y.  Note that y is used for scratch space and may be override
     * @param normType
     * @param x
     * @param y
     * @param norm, the norm must be the same size as the block size of x/y
     * @return
     */
    static PetscErrorCode ComputeNorm(Norm normType, Vec x, Vec y, PetscReal norm[]);

    MathUtilities() = delete;
};

/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::ostream& operator<<(std::ostream& os, const MathUtilities::Norm& v);
/**
 * Support function for the Scope Enum
 * @param os
 * @param v
 * @return
 */
std::istream& operator>>(std::istream& is, MathUtilities::Norm& v);

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MATHUTILITIES_HPP
