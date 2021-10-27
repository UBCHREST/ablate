#ifndef ABLATELIBRARY_FINITEELEMENT_HPP
#define ABLATELIBRARY_FINITEELEMENT_HPP

#include <solver/timeStepper.hpp>
#include <string>
#include <vector>
#include "boundaryConditions/boundaryCondition.hpp"
#include "eos/eos.hpp"
#include "fvSupport.h"
#include "mathFunctions/fieldFunction.hpp"
#include "solver/solver.hpp"

namespace ablate::finiteElement {

class FiniteElement : public solver::Solver, public solver::IFunction, public solver::BoundaryFunction {
   private:
    // helper function to register fv field
    void RegisterFiniteElementField(const domain::FieldDescriptor&);

    std::vector<domain::FieldDescriptor> fieldDescriptors;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization;
    const std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFieldsUpdaters;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;

    // Replacement calls for PETSC versions allowing multiple DS
    static PetscErrorCode DMPlexInsertBoundaryValues_Plex(DM dm, PetscDS ds, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM);
    static PetscErrorCode DMPlexInsertTimeDerivativeBoundaryValues_Plex(DM dm, PetscDS ds, PetscBool insertEssential, Vec locX, PetscReal time, Vec faceGeomFVM, Vec cellGeomFVM, Vec gradFVM);

   public:
    FiniteElement(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::vector<domain::FieldDescriptor> fieldDescriptors,
                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution);

    /** SubDomain Register and Setup **/
    void Register(std::shared_ptr<ablate::domain::SubDomain> subDomain) override;
    void Setup() override;
    void Initialize() override;

    virtual void CompleteFlowInitialization(DM, Vec) = 0;

    /**
     * function to update the aux fields.
     */
    static void UpdateAuxFields(TS ts, FiniteElement& fe);

    void Save(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) const override;

    /** Functions to compute F and and jacobian for the fintie element method over this subDomain/DS. **/
    PetscErrorCode ComputeIFunction(PetscReal time, Vec locX, Vec locX_t, Vec locF) override;
    PetscErrorCode ComputeIJacobian(PetscReal time, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP) override;
    PetscErrorCode ComputeBoundary(PetscReal time, Vec locX, Vec locX_t) override;
};
}  // namespace ablate::finiteElement

#endif  // ABLATELIBRARY_FINITEVOLUME_HPP
