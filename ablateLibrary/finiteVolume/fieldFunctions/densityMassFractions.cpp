#include "densityMassFractions.hpp"
#include <mathFunctions/functionPointer.hpp>

ablate::finiteVolume::fieldFunctions::DensityMassFractions::DensityMassFractions(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowStateIn)
    : ablate::mathFunctions::FieldFunction("densityYi",
                                           std::make_shared<ablate::mathFunctions::FunctionPointer>(ablate::finiteVolume::fieldFunctions::CompressibleFlowState::ComputeDensityYiFromState, flowStateIn.get())),
      flowState(flowStateIn) {}

#include "parser/registrar.hpp"
REGISTER(ablate::mathFunctions::FieldFunction, ablate::finiteVolume::fieldFunctions::DensityMassFractions, "initializes the densityYi conserved field variables based upon a CompressibleFlowState",
         ARG(ablate::finiteVolume::fieldFunctions::CompressibleFlowState, "state", "The CompressibleFlowState used to initialize"));