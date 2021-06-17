#ifndef ABLATELIBRARY_COMPRESSIBLEFLOW_H
#define ABLATELIBRARY_COMPRESSIBLEFLOW_H

#include <fvSupport.h>
#include <petsc.h>
#include <eos/eos.hpp>
#include <string>
#include "compressibleFlow.h"
#include "flow.hpp"
#include "flow/fluxDifferencer/fluxDifferencer.hpp"
#include "mesh/mesh.hpp"
#include "parameters/parameters.hpp"

namespace ablate::flow {
class CompressibleFlow : public Flow {
   private:
    std::shared_ptr<eos::EOS> eos;
    std::shared_ptr<fluxDifferencer::FluxDifferencer> fluxDifferencer;

    FlowData_CompressibleFlow compressibleFlowData;

    // functions to update each aux field
    FVAuxFieldUpdateFunction auxFieldUpdateFunctions[TOTAL_COMPRESSIBLE_AUX_COMPONENTS];

    // hold functions needed to update diffusion terms
    std::vector<FVDiffusionFunction> diffusionCalculationFunctions;

    // hold the update functions for source
    std::vector<FVMRHSFunctionDescription> rhsFunctionDescriptions;

    // static function to update the flowfield
    static void ComputeTimeStep(TS, Flow &);

   public:
    CompressibleFlow(std::string name, std::shared_ptr<mesh::Mesh> mesh, std::shared_ptr<eos::EOS> eos, std::shared_ptr<parameters::Parameters> parameters,
                     std::shared_ptr<fluxDifferencer::FluxDifferencer> = {}, std::shared_ptr<parameters::Parameters> options = {},
                     std::vector<std::shared_ptr<mathFunctions::FieldSolution>> initialization = {}, std::vector<std::shared_ptr<boundaryConditions::BoundaryCondition>> boundaryConditions = {},
                     std::vector<std::shared_ptr<mathFunctions::FieldSolution>> exactSolutions = {});
    ~CompressibleFlow() override;

    void CompleteProblemSetup(TS ts) override;
    void CompleteFlowInitialization(DM, Vec) override;
    static PetscErrorCode CompressibleFlowRHSFunctionLocal(DM dm, PetscReal time, Vec locXVec, Vec globFVec, void *ctx);
};
}  // namespace ablate::flow

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOW_H
