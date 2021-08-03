#include "massFractions.hpp"
#include <mathFunctions/functionPointer.hpp>

ablate::flow::fieldFunctions::MassFractions::MassFractions(std::shared_ptr<ablate::flow::fieldFunctions::CompressibleFlowState> flowStateIn)
    : ablate::mathFunctions::FieldFunction("densityYi",
                                           std::make_shared<ablate::mathFunctions::FunctionPointer>(ablate::flow::fieldFunctions::CompressibleFlowState::ComputeDensityYiFromState, flowStateIn.get())),
      flowState(flowStateIn) {}
