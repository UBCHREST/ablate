#ifndef ABLATELIBRARY_DENSITYVOLUMEFRACTION_HPP
#define ABLATELIBRARY_DENSITYVOLUMEFRACTION_HPP

#include <eos/eos.hpp>
#include <mathFunctions/fieldFunction.hpp>
#include "compressibleFlowState.hpp"

namespace ablate::finiteVolume::fieldFunctions {

class DensityVolumeFraction : public ablate::mathFunctions::FieldFunction {
   public:
    explicit DensityVolumeFraction(std::shared_ptr<ablate::finiteVolume::fieldFunctions::CompressibleFlowState> flowState, std::shared_ptr<ablate::domain::Region> region = {});
};

}  // namespace ablate::finiteVolume::fieldFunctions
#endif  // ABLATELIBRARY_DENSITYVOLUMEFRACTION_HPP
