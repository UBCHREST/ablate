#ifndef ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP
#define ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "eos/eos.hpp"
#include "parameters/mapParameters.hpp"

namespace ablate::finiteVolume {

class CompressibleFlowFields : public domain::FieldDescriptor {
   public:
    typedef enum { RHO, RHOE, RHOU, RHOV, RHOW } EulerComponents;

    inline const static std::string EULER_FIELD = "euler";
    inline const static std::string DENSITY_YI_FIELD = "densityYi";
    inline const static std::string YI_FIELD = "yi";
    inline const static std::string DENSITY_EV_FIELD = "densityEV";
    inline const static std::string EV_FIELD = "ev";
    inline const static std::string TEMPERATURE_FIELD = "temperature";
    inline const static std::string VELOCITY_FIELD = "velocity";
    inline const static std::string PRESSURE_FIELD = "pressure";

   private:
    const std::shared_ptr<eos::EOS> eos;
    const std::vector<std::string> extraVariables;
    const std::shared_ptr<domain::Region> region;
    const std::shared_ptr<parameters::Parameters> conservedFieldOptions;
    const std::shared_ptr<parameters::Parameters> auxFieldOptions = ablate::parameters::MapParameters::Create({{"petscfv_type", "leastsquares"}, {"petsclimiter_type", "none"}});

   public:
    CompressibleFlowFields(std::shared_ptr<eos::EOS>, std::vector<std::string> = {}, std::shared_ptr<domain::Region> = {}, std::shared_ptr<parameters::Parameters> conservedFieldParameters = {});

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP
