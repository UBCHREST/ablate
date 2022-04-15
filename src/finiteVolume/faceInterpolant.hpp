#ifndef ABLATELIBRARY_FACEINTERPOLANT_HPP
#define ABLATELIBRARY_FACEINTERPOLANT_HPP

#include <memory>
#include "domain/subDomain.hpp"

namespace ablate::finiteVolume {

class FaceInterpolant {
   private:
    //! use the subDomain to setup the problem
    std::shared_ptr<ablate::domain::SubDomain> subDomain;

    //! The region of this solver.
    const std::shared_ptr<domain::Region> region;

    //! store the aux and solution variable sizes for this domain
    PetscInt solTotalSize = 0;

    //! store the aux and solution variable sizes for this domain
    PetscInt auxTotalSize =0;

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
    static inline void AddToArray(I size,const T* input, T* sum, T factor) {
        for (I d = 0; d < size; d++) {
            sum[d] += factor*input[d];
        }
    }

   public:
    FaceInterpolant(std::shared_ptr<ablate::domain::SubDomain> subDomain, std::shared_ptr<domain::Region> region, Vec faceGeomVec, Vec cellGeomVec);
    ~FaceInterpolant();

    void GetInterpolatedFaceVectors(Vec solutionVec, Vec auxVec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec );

    void RestoreInterpolatedFaceVectors(Vec solutionVec, Vec auxVec, Vec& faceSolutionVec, Vec& faceAuxVec, Vec& faceSolutionGradVec, Vec& faceAuxGradVec );
};

}  // namespace ablate::finiteVolume
#endif  // ABLATELIBRARY_FACEINTERPOLANT_HPP
