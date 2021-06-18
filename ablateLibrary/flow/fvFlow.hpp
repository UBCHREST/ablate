#ifndef ABLATELIBRARY_FVFLOW_HPP
#define ABLATELIBRARY_FVFLOW_HPP

#include "flow.hpp"
#include <fvSupport.h>


namespace ablate::flow {

class FVFlow : public Flow {
   protected: // move this to private
    // hold the update functions for source
    std::vector<FVMRHSFunctionDescription> rhsFunctionDescriptions;

    // functions to update each aux field
    std::vector<FVAuxFieldUpdateFunction> auxFieldUpdateFunctions;
    std::vector<void*> auxFieldUpdateContexts;


   public:
    FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
        std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
        std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution);
    ~FVFlow() override = default;

    void CompleteProblemSetup(TS ts) override;

    static PetscErrorCode FVRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx);
};

}
#endif  // ABLATELIBRARY_FVFLOW_HPP
