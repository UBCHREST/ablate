#ifndef ABLATELIBRARY_CELLSOLVER_HPP
#define ABLATELIBRARY_CELLSOLVER_HPP

#include <petsc.h>
#include <functional>
#include <vector>
#include "solver.hpp"

namespace ablate::solver {

class CellSolver : public solver::Solver {
   public:
    using AuxFieldUpdateFunction = PetscErrorCode (*)(PetscReal time, PetscInt dim, const PetscFVCellGeom* cellGeom, const PetscInt uOff[], const PetscScalar* u, const PetscInt aOff[],
                                                      PetscScalar* auxField, void* ctx);

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

    std::vector<AuxFieldUpdateFunctionDescription> auxFieldUpdateFunctionDescriptions;

   protected:
    //! Vector used to describe the entire cell geom of the dm.  This is constant and does not depend upon region.
    Vec cellGeomVec = nullptr;

    //! Vector used to describe the entire face geom of the dm.  This is constant and does not depend upon region.
    Vec faceGeomVec = nullptr;

   public:
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
     * Helper function to march over each cell and update the aux Fields
     * @param flow
     * @param time
     * @param locXVec
     * @param updateFunction
     */
    void UpdateAuxFields(PetscReal time, Vec locXVec, Vec locAuxField);

    /**
     * Setup the subdomain cell solver vectors
     */
    void Setup() override;
};
}  // namespace ablate::solver

#endif  // ABLATELIBRARY_CELLSOLVER_HPP
