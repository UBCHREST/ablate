#ifndef ABLATELIBRARY_LEASTSQUARES_HPP
#define ABLATELIBRARY_LEASTSQUARES_HPP

#include "domain/subDomain.hpp"
#include "faceStencilGenerator.hpp"
namespace ablate::finiteVolume::stencil {

/**
 * computes the weight/gradient weights assuming least squares directly to the face
 */
class LeastSquares : public FaceStencilGenerator {
   private:
    // local scratch variables
    PetscInt maxFaces = 0;
    std::vector<PetscScalar> dx;
    PetscFV gradientCalculator = nullptr;

   public:
    void Generate(PetscInt face, Stencil& stencil, const domain::SubDomain& subDomain, const std::shared_ptr<domain::Region> solverRegion, DM cellDM, const PetscScalar* cellGeomArray, DM faceDM,
                  const PetscScalar* faceGeomArray) override;
    ~LeastSquares() override;
};

}  // namespace ablate::finiteVolume::stencil
#endif  // ABLATELIBRARY_LEASTSQUARES_HPP
