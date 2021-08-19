#include "euler.hpp"
#include <mathFunctions/functionPointer.hpp>

ablate::flow::fieldFunctions::Euler::Euler(std::shared_ptr<ablate::flow::fieldFunctions::CompressibleFlowState> flowStateIn)
    : ablate::mathFunctions::FieldFunction("euler",
                                           std::make_shared<ablate::mathFunctions::FunctionPointer>(ablate::flow::fieldFunctions::CompressibleFlowState::ComputeEulerFromState, flowStateIn.get())),
      flowState(flowStateIn) {}

#include "parser/registrar.hpp"
REGISTER(ablate::mathFunctions::FieldFunction, ablate::flow::fieldFunctions::Euler, "initializes the euler conserved field variables based upon a CompressibleFlowState",
         ARG(ablate::flow::fieldFunctions::CompressibleFlowState, "state", "The CompressibleFlowState used to initialize"));
