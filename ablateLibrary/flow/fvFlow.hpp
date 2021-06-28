#ifndef ABLATELIBRARY_FVFLOW_HPP
#define ABLATELIBRARY_FVFLOW_HPP

#include <fvSupport.h>
#include <string>
#include <vector>
#include "flow.hpp"

namespace ablate::flow {

class FVFlow : public Flow {
   private:  // move this to private
    // hold the update functions for flux and point sources
    std::vector<FVMRHSFluxFunctionDescription> rhsFluxFunctionDescriptions;
    std::vector<FVMRHSPointFunctionDescription> rhsPointFunctionDescriptions;

    // functions to update each aux field
    std::vector<FVAuxFieldUpdateFunction> auxFieldUpdateFunctions;
    std::vector<void*> auxFieldUpdateContexts;

   public:
    FVFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<parameters::Parameters> parameters, std::shared_ptr<parameters::Parameters> options,
           std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions,
           std::vector<std::shared_ptr<mathFunctions::FieldSolution>> auxiliaryFields, std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolution);
    ~FVFlow() override = default;

    void CompleteProblemSetup(TS ts) override;

    static PetscErrorCode FVRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void* ctx);

    /**
     * Register a FVM rhs source flux function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(FVMRHSFluxFunction function, void* context, std::string field, std::vector<std::string> inputFields, std::vector<std::string> auxFields);

    /**
     * Register a FVM rhs point function
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterRHSFunction(FVMRHSPointFunction function, void* context, std::string field, std::vector<std::string> inputFields, std::vector<std::string> auxFields);

    /**
     * Register a auxFieldUpdate
     * @param function
     * @param context
     * @param field
     * @param inputFields
     * @param auxFields
     */
    void RegisterAuxFieldUpdate(FVAuxFieldUpdateFunction function, void* context, std::string auxField);
};

}  // namespace ablate::flow
#endif  // ABLATELIBRARY_FVFLOW_HPP
