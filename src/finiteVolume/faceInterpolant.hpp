#ifndef ABLATELIBRARY_FACEINTERPOLANT_HPP
#define ABLATELIBRARY_FACEINTERPOLANT_HPP

#include <memory>
#include "domain/range.hpp"
#include "domain/subDomain.hpp"
#include "stencils/stencil.hpp"

namespace ablate::finiteVolume {

class FaceInterpolant {
   private:
    //! use the subDomain to setup the problem
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    //! store the aux and solution variable sizes for this domain
    PetscInt solTotalSize = 0;

    //! store the aux and solution variable sizes for this domain
    PetscInt auxTotalSize = 0;

    //! store the global face start to compute stencil location
    PetscInt globalFaceStart = -1;

    /**
     * Create a dm for all face values
     */
    DM faceSolutionDm = nullptr;

    /**
     * Create a dm for all face values and gradients
     */
    DM faceSolutionGradDm = nullptr;

    /**
     * Create a dm for all face values
     */
    DM faceAuxDm = nullptr;

    /**
     * Create a dm for all face values and gradients
     */
    DM faceAuxGradDm = nullptr;

    /**
     * Helper function to create a faceDM
     * @param totalDim
     * @param dm
     * @param newDm
     */
    static void CreateFaceDm(PetscInt totalDim, DM dm, DM& newDm);

    /**
     * Store the interpolant for every face
     */
    std::vector<stencil::Stencil> stencils;

    template <class I, class T>
    static inline void AddToArray(I size, const T* input, T* sum, T factor) {
        for (I d = 0; d < size; d++) {
            sum[d] += factor * input[d];
        }
    }

   public:
    /**
     *
     * @param subDomain
     * @param faceGeomVec
     * @param cellGeomVec
     */
    FaceInterpolant(const std::shared_ptr<ablate::domain::SubDomain>& subDomain, const std::shared_ptr<domain::Region> solverRegion, Vec faceGeomVec, Vec cellGeomVec);
    ~FaceInterpolant();

    /**
     * Function assumes that the left/right solution and aux variables are continuous across the interface and values are interpolated to the face
     */
    using ContinuousFluxFunction = PetscErrorCode (*)(PetscInt dim, const PetscFVFaceGeom* fg, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar field[], const PetscScalar grad[],
                                                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[], const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * struct to describe how to compute RHS finite volume flux source terms with a continuous field
     */
    struct ContinuousFluxFunctionDescription {
        ContinuousFluxFunction function;
        void* context;

        std::vector<PetscInt> updateFields;
        std::vector<PetscInt> inputFields;
        std::vector<PetscInt> auxFields;
    };

    /**
     * Adds in contributions for face based rhs functions
     * @param time
     * @param locXVec
     * @param locFVec
     */
    void ComputeRHS(PetscReal time, Vec locXVec, Vec locAuxVec, Vec locFVec, const std::shared_ptr<domain::Region>& solverRegion,
                    std::vector<FaceInterpolant::ContinuousFluxFunctionDescription>& rhsFunctions, const ablate::domain::Range& faceRange, Vec cellGeomVec, Vec faceGeomVec);

    /**
     * function to get the interpolated values on the face
     * @param solutionVec
     * @param auxVec
     * @param faceSolutionVec
     * @param faceAuxVec
     * @param faceSolutionGradVec
     * @param faceAuxGradVec
     */
    void GetInterpolatedFaceVectors(Vec solutionVec, Vec auxVec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec);

    /**
     * function to return the interpolated values on the face
     * @param solutionVec
     * @param auxVec
     * @param faceSolutionVec
     * @param faceAuxVec
     * @param faceSolutionGradVec
     * @param faceAuxGradVec
     */
    void RestoreInterpolatedFaceVectors(Vec solutionVec, Vec auxVec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec);
};

}  // namespace ablate::finiteVolume
#endif  // ABLATELIBRARY_FACEINTERPOLANT_HPP
