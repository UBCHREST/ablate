#include "euler.hpp"
#include <mathFunctions/functionPointer.hpp>

ablate::finiteVolume::fieldFunctions::Euler::Euler(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowStateIn)
    : ablate::mathFunctions::FieldFunction("euler",
                                           std::make_shared<ablate::mathFunctions::FunctionPointer>(ablate::finiteVolume::fieldFunctions::CompressibleFlowState::ComputeEulerFromState, flowStateIn.get())),
      flowState(flowStateIn) {}

#include "parser/registrar.hpp"
REGISTER(ablate::mathFunctions::FieldFunction, ablate::finiteVolume::fieldFunctions::Euler, "initializes the euler conserved field variables based upon a CompressibleFlowState",
         ARG(ablate::finiteVolume::fieldFunctions::CompressibleFlowState, "state", "The CompressibleFlowState used to initialize"));
