#ifndef ABLATELIBRARY_FACEINTERPOLANT_HPP
#define ABLATELIBRARY_FACEINTERPOLANT_HPP

#include <memory>
#include "domain/subDomain.hpp"

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
    PetscInt globalFaceStart;

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
     * struct to hold the gradient stencil for the boundary
     */
    struct Stencil {
        PetscInt faceId;

        /** Area-scaled normals */
        PetscReal area[3];
        /** normal **/
        PetscReal normal[3];
        /** Location of centroid*/
        PetscReal centroid[3];

        /** store the stencil size for easy access */
        PetscInt stencilSize;
        /** The points in the stencil*/
        std::vector<PetscInt> stencil;
        /** The weights in [point*dim + dir] order */
        std::vector<PetscScalar> weights;
        /** The gradient weights in [point*dim + dir] order */
        std::vector<PetscScalar> gradientWeights;
    };

    /**
     * Store the interpolant for every face
     */
    std::vector<Stencil> stencils;

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
    FaceInterpolant(std::shared_ptr<ablate::domain::SubDomain> subDomain, Vec faceGeomVec, Vec cellGeomVec);
    ~FaceInterpolant();

    /**
     * Function assumes that the left/right solution and aux variables are continuous across the interface and values are interpolated to the face
     */
    using ContinuousFluxFunction = PetscErrorCode (*)(PetscInt dim, const PetscReal* area, const PetscReal* normal, const PetscReal* centroid, const PetscInt uOff[], const PetscInt uOff_x[],
                                                      const PetscScalar field[], const PetscScalar grad[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar aux[],
                                                      const PetscScalar gradAux[], PetscScalar flux[], void* ctx);

    /**
     * struct to describe how to compute RHS finite volume flux source terms with a continuous field
     */
    struct ContinuousFluxFunctionDescription {
        ContinuousFluxFunction function;
        void* context;

        PetscInt field;
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
                    std::vector<FaceInterpolant::ContinuousFluxFunctionDescription>& rhsFunctions, PetscInt fStart, PetscInt fEnd, const PetscInt* faces, Vec cellGeomVec);

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
