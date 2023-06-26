#ifndef ABLATELIBRARY_CELLSOLVER_HPP
#define ABLATELIBRARY_CELLSOLVER_HPP

#include <petsc.h>
#include <functional>
#include <vector>
#include "solver.hpp"

namespace ablate::solver {

class CellSolver : public solver::Solver {
   public:
    //! function template for updating the aux field
    using AuxFieldUpdateFunction = PetscErrorCode (*)(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* u, const PetscInt aOff[],
                                                      PetscScalar* auxField, void* ctx);

    //! function template for updating the solution field
    using SolutionFieldUpdateFunction = PetscErrorCode (*)(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], PetscScalar* u, void* ctx);

   private:
    /**
     * struct to describe how to compute the aux variable update
     */
    struct AuxFieldUpdateFunctionDescription {
        AuxFieldUpdateFunction function;
        void* context;
        std::vector<PetscInt> inputFields;
        std::vector<PetscInt> auxFields;
    };

    //! list of auxField update functions
    std::vector<AuxFieldUpdateFunctionDescription> auxFieldUpdateFunctionDescriptions;

    /**
     * struct to describe how to compute the solution variable update
     */
    struct SolutionFieldUpdateFunctionDescription {
        SolutionFieldUpdateFunction function;
        void* context;
        std::vector<PetscInt> inputFieldsOffsets;
    };

    //! list of auxField update functions
    std::vector<SolutionFieldUpdateFunctionDescription> solutionFieldUpdateFunctionDescriptions;

   protected:
    //! Vector used to describe the entire cell geom of the dm.  This is constant and does not depend upon region.
    Vec cellGeomVec = nullptr;

    //! Vector used to describe the entire face geom of the dm.  This is constant and does not depend upon region.
    Vec faceGeomVec = nullptr;

   public:
    /**
     * Create a base solver used for cell based solvers
     * @param solverId
     * @param options
     */
    explicit CellSolver(std::string solverId, std::shared_ptr<domain::Region> = {}, std::shared_ptr<parameters::Parameters> options = nullptr);
    ~CellSolver() override;

    /**
     * Register a auxFieldUpdate
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterAuxFieldUpdate(AuxFieldUpdateFunction function, void* context, const std::vector<std::string>& auxField, const std::vector<std::string>& inputFields);

    /**
     * Register a auxFieldUpdate
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterSolutionFieldUpdate(SolutionFieldUpdateFunction function, void* context, const std::vector<std::string>& inputFields);

    /**
     * Helper function to march over each cell and update the aux Fields
     * @param time
     * @param locXVec
     * @param locAuxField
     */
    void UpdateAuxFields(PetscReal time, Vec locXVec, Vec locAuxField);

    /**
     * Helper function to march over each cell and update the solution fields if specified (including ghost nodes)
     * @param flow
     * @param locXVec
     */
    void UpdateSolutionFields(PetscReal time, Vec globXVec);

    /**
     * Setup the subdomain cell solver vectors
     */
    void Setup() override;

    /**
     * Call any solution update functions before the first timestep
     */
    void Initialize() override;
};
}  // namespace ablate::solver

#endif  // ABLATELIBRARY_CELLSOLVER_HPP
