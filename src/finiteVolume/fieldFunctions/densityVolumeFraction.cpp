#include "densityVolumeFraction.hpp"
#include "finiteVolume/processes/twoPhaseEulerAdvection.hpp"

ablate::finiteVolume::fieldFunctions::DensityVolumeFraction::DensityVolumeFraction(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowStateIn,
                                                                                   std::shared_ptr<ablate::domain::Region> region)
    : ablate::mathFunctions::FieldFunction(processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD, flowStateIn->GetFieldFunction(processes::TwoPhaseEulerAdvection::DENSITY_VF_FIELD), {}, region) {}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::FieldFunction, ablate::finiteVolume::fieldFunctions::DensityVolumeFraction, "initializes the densityVF conserved field for two phase flow based upon a CompressibleFlowState",
         ARG(ablate::finiteVolume::fieldFunctions::CompressibleFlowState, "state", "The CompressibleFlowState used to initalize"),
         OPT(ablate::domain::Region, "region", "A subset of the domain to apply the field function"));
