#ifndef ABLATELIBRARY_FIELD_FUNCTION_DENSITY_PROGRESSVARIABLE_HPP
#define ABLATELIBRARY_FIELD_FUNCTION_DENSITY_PROGRESSVARIABLE_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"

namespace ablate::finiteVolume::fieldFunctions {

class DensityProgressVariables : public ablate::mathFunctions::FieldFunction {
   public:
    explicit DensityProgressVariables(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowState, std::shared_ptr<ablate::domain::Region> region = {});
};

}  // namespace ablate::finiteVolume::fieldFunctions
#endif  // ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
