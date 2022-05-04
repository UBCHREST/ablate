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

   public:
    /**
     *
     * @param solverId the id for this solver
     * @param region the boundary cell region
     * @param rayNumber
     * @param options other options
     */
    RadiationSolver(std::string solverId, std::shared_ptr<domain::Region> region, int rayNumber, std::shared_ptr<parameters::Parameters> options);
    ~RadiationSolver() override;

    /** SubDomain Register and Setup **/
    void Setup() override;
    void Initialize() override;

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

    PetscErrorCode RayProduct(PetscReal time, PetscInt segSteps);

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
    PetscInt rayNumber;

    PetscInt nSteps = 100; //number of steps that each ray will go through //This won't be used
    PetscReal h = 0.02; //This is the DEFAULT step size which should be set by the user input
    PetscInt nTheta = rayNumber; //The DEFAULT number of angles to solve with, should be given by user input probably?
    PetscInt nPhi = 1;//2*rayNumber; //The DEFAULT number of angles to solve with, should be given by user input

    std::vector<std::vector<std::vector<std::vector<PetscInt>>>> rays;//(std::vector<std::vector<std::vector<PetscInt>>>()); //Indices: Cell, angle (theta), angle(phi), space steps
    //PetscReal radGain;
    //PetscViewer viewer;
    Vec origin;
};

}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_BOUNDARYSOLVER_HPP
