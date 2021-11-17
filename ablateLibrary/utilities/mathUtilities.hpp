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

    /**
     * Computes the transformation matrix from the global Cartesian system to a normal coordinate system.  The first component of the transformed system
     * is always the normal direction
     * @param dim
     * @param normal
     * @param transformationMatrix [row, col]
     */
    void ComputeTransformationMatrix(PetscInt dim, const PetscScalar normal[3], PetscScalar transformationMatrix[3][3]);

   private:
    MathUtilities() = delete;
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MATHUTILITIES_HPP
