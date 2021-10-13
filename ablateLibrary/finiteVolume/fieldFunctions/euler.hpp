#ifndef ABLATELIBRARY_FIELD_FUNCTION_EULER_HPP
#define ABLATELIBRARY_FIELD_FUNCTION_EULER_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"

namespace ablate::finiteVolume::fieldFunctions {

class Euler : public ablate::mathFunctions::FieldFunction {
   private:
    const std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowState;

   public:
    explicit Euler(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowState);
};

}  // namespace ablate::flow::fieldFunctions
#endif  // ABLATELIBRARY_FIELD_SOLUTION_EULER_HPP
