#ifndef ABLATELIBRARY_FIELD_FUNCTION_DENSITY_MASSFRACTIONS_HPP
#define ABLATELIBRARY_FIELD_FUNCTION_DENSITY_MASSFRACTIONS_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"

namespace ablate::flow::fieldFunctions {

class DensityMassFractions : public ablate::mathFunctions::FieldFunction {
   private:
    const std::shared_ptr<ablate::flow::fieldFunctions::CompressibleFlowState> flowState;

   public:
    explicit DensityMassFractions(std::shared_ptr<ablate::flow::fieldFunctions::CompressibleFlowState> flowState);
};

}  // namespace ablate::flow::fieldFunctions
#endif  // ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
