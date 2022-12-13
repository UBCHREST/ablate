#ifndef ABLATELIBRARY_TURBULENCEFLOWFIELDS_HPP
#define ABLATELIBRARY_TURBULENCEFLOWFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "compressibleFlowFields.hpp"
#include "domain/fieldDescriptor.hpp"
#include "eos/eos.hpp"
#include "parameters/mapParameters.hpp"

namespace ablate::finiteVolume {

/**
 * Used to create the required flow fields to model turbulence
 */
class TurbulenceFlowFields : public domain::FieldDescriptor {
   public:
    //! the conserved (density*tke) solution field for species mass fractions
    inline const static std::string TKE_FIELD = "tke";
    inline const static std::string DENSITY_TKE_FIELD = CompressibleFlowFields::CONSERVED + TKE_FIELD;

   private:
    const std::shared_ptr<domain::Region> region;
    const std::shared_ptr<parameters::Parameters> conservedFieldOptions;
    const std::shared_ptr<parameters::Parameters> auxFieldOptions = ablate::parameters::MapParameters::Create({{"petscfv_type", "leastsquares"}, {"petsclimiter_type", "none"}});

   public:
    explicit TurbulenceFlowFields(std::shared_ptr<domain::Region> = {}, std::shared_ptr<parameters::Parameters> conservedFieldParameters = {});

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP
