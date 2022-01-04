#ifndef ABLATELIBRARY_FIELD_FUNCTION_DENSITY_MASSFRACTIONS_HPP
#define ABLATELIBRARY_FIELD_FUNCTION_DENSITY_MASSFRACTIONS_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"

namespace ablate::finiteVolume::fieldFunctions {

class DensityMassFractions : public ablate::mathFunctions::FieldFunction {
   private:
    const std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowState;

   public:
    explicit DensityMassFractions(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowState, std::shared_ptr<ablate::domain::Region> region = {});
};

}  // namespace ablate::finiteVolume::fieldFunctions
#endif  // ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
