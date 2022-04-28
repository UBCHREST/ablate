#ifndef ABLATELIBRARY_BOUNDARYSOLVER_HPP
#define ABLATELIBRARY_BOUNDARYSOLVER_HPP

#include <memory>
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::boundarySolver {

// forward declare the boundaryProcess
class BoundaryProcess;

class BoundarySolver : public solver::CellSolver, public solver::RHSFunction {
   public:
    /**
     * Boundary information.
     */
    typedef struct {
        PetscReal normal[3];   /* normals (pointing into the boundary from the other region) */
        PetscReal areas[3];    /* Area-scaled normals */
        PetscReal centroid[3]; /* Location of centroid (quadrature point) */
    } BoundaryFVFaceGeom;

    using BoundarySourceFunction = PetscErrorCode (*)(PetscInt dim, const BoundaryFVFaceGeom* fg, const PetscFVCellGeom* boundaryCell, const PetscInt uOff[], const PetscScalar* boundaryValues,
                                                      const PetscScalar* stencilValues[], const PetscInt aOff[], const PetscScalar* auxValues, const PetscScalar* stencilAuxValues[],
                                                      PetscInt stencilSize, const PetscInt stencil[], const PetscScalar stencilWeights[], const PetscInt sOff[], PetscScalar source[], void* ctx);

    /**
     * Boundaries can be treated in two different ways, point source on the boundary or distributed in the other phase.  For the Distributed model, the source is divided by volume in each case
     */
    enum class BoundarySourceType {
        Point, /** the source terms are added to boundary cell **/
        Distributed//,  /** the source terms are distributed to neighbor cells based upon the stencil **/
//        Flux /** the source term are added to only one neighbor cell.  Each cell faces are not merged **/
    };

    /**
     * public helper function to compute the gradient
     * @param dim
     * @param boundaryValue
     * @param stencilValues
     * @param stencilWeights
     * @param grad
     */
    static void ComputeGradient(PetscInt dim, PetscScalar boundaryValue, PetscInt stencilSize, const PetscScalar* stencilValues, const PetscScalar* stencilWeights, PetscScalar* grad);

    /**
     * public helper function to compute dPhiDNorm
     * @param dim
     * @param boundaryValue
     * @param stencilValues
     * @param stencilWeights
     * @param grad
     */
    static void ComputeGradientAlongNormal(PetscInt dim, const BoundaryFVFaceGeom* fg, PetscScalar boundaryValue, PetscInt stencilSize, const PetscScalar* stencilValues,
                                           const PetscScalar* stencilWeights, PetscScalar& dPhiDNorm);

   private:
    /**
     * struct to hold the gradient stencil for the boundary
     */
    struct GradientStencil {
        /** the boundary cell for this stencil **/
        PetscInt cellId;
        /** the boundary geom for this stencil **/
        BoundaryFVFaceGeom geometry;
        /** The points in the stencil*/
        std::vector<PetscInt> stencil;
        /** The weights in [point*dim + dir] order */
        std::vector<PetscScalar> gradientWeights;
        /** store the stencil size for easy access */
        PetscInt stencilSize;
        /** The distribution weights in order */
        std::vector<PetscScalar> distributionWeights;
        /** Store the volume for each stencil cell */
        std::vector<PetscScalar> volumes;
    };

    /**
     * struct to describe how to compute the source terms for boundary
     */
    struct BoundaryFunctionDescription {
        BoundarySourceFunction function;
        void* context;
        BoundarySourceType type;

        std::vector<PetscInt> sourceFields;
        std::vector<PetscInt> inputFields;
        std::vector<PetscInt> auxFields;
    };

    // Hold the region used to define the boundary faces
    const std::shared_ptr<domain::Region> fieldBoundary;

    // hold the update functions for flux and point sources
    std::vector<BoundaryFunctionDescription> boundaryFunctions;

    // Hold a list of boundaryProcesses that contribute to this solver
    std::vector<std::shared_ptr<BoundaryProcess>> boundaryProcesses;

    // Hold a list of GradientStencils, this order corresponds to the face order
    std::vector<GradientStencil> gradientStencils;

    // keep track of maximumStencilSize
    PetscInt maximumStencilSize = 0;

    // The PetscFV (usually the least squares method) is used to compute the gradient weights
    PetscFV gradientCalculator = nullptr;

   public:
    /**
     *
     * @param solverId the id for this solver
     * @param region the boundary cell region
     * @param fieldBoundary the region describing the faces between the boundary and field
     * @param boundaryProcesses a list of boundary processes
     * @param options other options
     */
    BoundarySolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary, std::vector<std::shared_ptr<BoundaryProcess>> boundaryProcesses,
                   std::shared_ptr<parameters::Parameters> options);
    ~BoundarySolver() override;

    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;

    /**
     * Register an arbitrary function.  The user is responsible for all work
     * @param function
     * @param context
     */
    void RegisterFunction(BoundarySourceFunction function, void* context, const std::vector<std::string>& sourceFields, const std::vector<std::string>& inputFields,
                          const std::vector<std::string>& auxFields, BoundarySourceType type = BoundarySourceType::Point);

    /**
     * Function passed into PETSc to compute the FV RHS
     * @param dm
     * @param time
     * @param locXVec
     * @param globFVec
     * @param ctx
     * @return
     */
    PetscErrorCode ComputeRHSFunction(PetscReal time, Vec locXVec, Vec locFVec) override;

    /**
     * Helper function to project values to a cell boundary instead of the cell centroid
     */
    void InsertFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initialization, PetscReal time = 0.0);

    /**
     * Return a reference to the boundary geometry.  This is a slow call and should only be done for init/debugging/testing
     */
    std::vector<GradientStencil>::const_iterator GetBoundaryGeometry(PetscInt cell) const;
};

}  // namespace ablate::boundarySolver
#endif  // ABLATELIBRARY_BOUNDARYSOLVER_HPP
