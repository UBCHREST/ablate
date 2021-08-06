#ifndef ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
#define ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"

namespace ablate::flow::fieldFunctions {

class MassFractions : public ablate::mathFunctions::FieldFunction {
   private:
    const std::shared_ptr<ablate::flow::fieldFunctions::CompressibleFlowState> flowState;

   public:
    explicit MassFractions(std::shared_ptr<ablate::flow::fieldFunctions::CompressibleFlowState> flowState);
};

}  // namespace ablate::flow::fieldFunctions
#endif  // ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
