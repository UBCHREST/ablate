#ifndef ABLATELIBRARY_LEASTSQUARESAVERAGEe_HPP
#define ABLATELIBRARY_LEASTSQUARESAVERAGEe_HPP

#include "domain/subDomain.hpp"
#include "faceStencilGenerator.hpp"
namespace ablate::finiteVolume::stencil {

/**
 * computes the weight/gradient weights assuming least squares directly to the face
 */
class LeastSquaresAverage : public FaceStencilGenerator {
   private:
    // local scratch variables
    PetscInt maxFaces = 0;
    std::vector<PetscScalar> dx;
    PetscFV gradientCalculator = nullptr;

   public:
    void Generate(PetscInt face, Stencil& stencil, const domain::SubDomain& subDomain, const std::shared_ptr<domain::Region> solverRegion, DM cellDM, const PetscScalar* cellGeomArray, DM faceDM,
                  const PetscScalar* faceGeomArray) override;
    ~LeastSquaresAverage() override;

    /**
     * helper function to compute left or right stencil
     */
    void ComputeNeighborCellStencil(PetscInt cell, Stencil& stencil, const domain::SubDomain& subDomain, const std::shared_ptr<domain::Region> solverRegion, DM cellDM,
                                    const PetscScalar* cellGeomArray, DM faceDM, const PetscScalar* faceGeomArray);
};

}  // namespace ablate::finiteVolume::stencil
#endif  // ABLATELIBRARY_LEASTSQUARES_HPP
