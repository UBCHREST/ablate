//
// Created by owen on 3/19/22.
//
#ifndef ABLATELIBRARY_RADIATION_HPP
#define ABLATELIBRARY_RADIATION_HPP

#include <memory>
#include <set>
#include "solver/cellSolver.hpp"
#include "solver/timeStepper.hpp"

namespace ablate::radiation {

class RadiationSolver : public solver::CellSolver, public solver::RHSFunction {  //Cell solver provides cell based functionality, right hand side function compatibility with finite element/ volume
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
    enum class BoundarySourceType { Point, Distributed };

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
     * @param radiationProcesses a list of boundary processes
     * @param options other options
     */
    RadiationSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options);
    ~RadiationSolver() override;

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
     *
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
    const BoundaryFVFaceGeom& GetBoundaryGeometry(PetscInt cell) const;

    ///Starting added radiation stuff here

    ///Class Methods
    void RayTrace(PetscReal time);
    static PetscReal FlameIntensity(PetscReal epsilon, PetscReal temperature);
    void RayInit();
    static PetscReal CSimp(PetscReal a, PetscReal b, std::vector<double> f);
    static PetscReal ReallySolveParallelPlates(PetscReal z); //std::vector<PetscReal>
    static PetscReal EInteg(int order, double x);

    ///Class Constants
    const PetscReal sbc = 5.6696e-8;  // Stefan-Boltzman Constant (J/K)
    const PetscReal refTemp = 298.15;
    const PetscReal pi = 3.1415926535897932384626433832795028841971693993;

    ///Class inputs and Variables
    DM vdm; //Abstract PETSc object that manages an abstract grid object and its interactions with the algebraic solvers
    Vec loctemp;
    IS vis;
    std::set<PetscInt> stencilSet;

    PetscInt dim; //Number of dimensions that the domain exists within

    PetscInt nSteps = 100; //number of steps that each ray will go through //This won't be used
    PetscReal h = 0.02; //This is the DEFAULT step size which should be set by the user input
    PetscInt nTheta = 10; //The DEFAULT number of angles to solve with, should be given by user input
    PetscInt nPhi = 5; //The DEFAULT number of angles to solve with, should be given by user input

    std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rays;//(std::vector<std::vector<std::vector<PetscInt>>>()); //Indices: Cell, angle (theta), angle(phi), space steps
    //PetscReal radGain;
    //PetscViewer viewer;
    Vec origin;
};

}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_BOUNDARYSOLVER_HPP
