#ifndef ABLATELIBRARY_BOUNDARYARBITRARYSOURCE_HPP
#define ABLATELIBRARY_BOUNDARYARBITRARYSOURCE_HPP

#include "boundarySolver/boundaryProcess.hpp"
#include "eos/transport/transportModel.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "finiteVolume/processes/pressureGradientScaling.hpp"
namespace ablate::boundarySolver::physics {

/**
 * produces required source terms in the "gas phase" assuming that the solid phase sublimates and no regression compared to the simulation time
 */
class ArbitrarySource : public BoundaryProcess {
   private:
    //! the arbitrary source function for each component
    const std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions;

    //! Store the boundary type
    const BoundarySolver::BoundarySourceType boundarySourceType;

    /** function to compute the arbitrary source term at each face **/
    static PetscErrorCode ArbitrarySourceFunction(PetscInt dim, const ablate::boundarySolver::BoundarySolver::BoundaryFVFaceGeom *fg, const PetscFVCellGeom *boundaryCell, const PetscInt uOff[],
                                                  const PetscScalar *boundaryValues, const PetscScalar *stencilValues[], const PetscInt aOff[], const PetscScalar *auxValues,
                                                  const PetscScalar *stencilAuxValues[], PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[],
                                                  PetscScalar source[], void *ctx);

    //! current simulation time
    PetscReal currentTime = 0.0;

   public:
    explicit ArbitrarySource(std::map<std::string, std::shared_ptr<ablate::mathFunctions::MathFunction>> functions, BoundarySolver::BoundarySourceType boundarySourceType);

    /**
     * Initialize the source function
     * @param bSolver
     */
    void Initialize(ablate::boundarySolver::BoundarySolver &bSolver) override;
};

}  // namespace ablate::boundarySolver::physics

#endif  // ABLATELIBRARY_SUBLIMATION_HPP
