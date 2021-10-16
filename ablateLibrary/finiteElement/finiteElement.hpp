#ifndef ABLATELIBRARY_FINITEELEMENT_HPP
#define ABLATELIBRARY_FINITEELEMENT_HPP

#include <solver/timeStepper.hpp>
#include <string>
#include <vector>
#include "finiteVolume/boundaryConditions/boundaryCondition.hpp"
#include "eos/eos.hpp"
#include "fvSupport.h"
#include "mathFunctions/fieldFunction.hpp"
#include "solver/solver.hpp"

namespace ablate::finiteElement{

class FiniteElement : public solver::Solver {
   private:

    // helper function to register fv field
    void RegisterFiniteElementField(const domain::FieldDescriptor&);

    std::vector<domain::FieldDescriptor> fieldDescriptors;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization;
    const std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>> boundaryConditions;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFieldsUpdaters;
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions;

   public:
    FiniteElement(std::string name, std::shared_ptr<parameters::Parameters> options,
                 std::vector<domain::FieldDescriptor> fieldDescriptors,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> initialization, std::vector<std::shared_ptr<finiteVolume::boundaryConditions::BoundaryCondition>> boundaryConditions,
                 std::vector<std::shared_ptr<mathFunctions::FieldFunction>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolution);

    void SetupDomain(std::shared_ptr<ablate::domain::SubDomain> subDomain) override;
    virtual void SetupElementDomain() = 0;
    void CompleteSetup(TS ts) override;

    virtual void CompleteFlowInitialization(DM, Vec) =0;

    /**
     * function to update the aux fields.
     */
    static void UpdateAuxFields(TS ts, FiniteElement& fe);

};
}

#endif  // ABLATELIBRARY_FINITEVOLUME_HPP
