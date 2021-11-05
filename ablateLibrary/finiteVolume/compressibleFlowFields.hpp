#ifndef ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP
#define ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "eos/eos.hpp"

namespace ablate::finiteVolume {

class CompressibleFlowFields: public domain::FieldDescriptor {
   public:
    inline const static std::string EULER_FIELD = "euler";
    inline const static std::string DENSITY_YI_FIELD = "densityYi";
    inline const static std::string YI_FIELD = "yi";
    inline const static std::string DENSITY_EV_FIELD = "densityEV";
    inline const static std::string EV_FIELD = "ev";
    inline const static std::string TEMPERATURE_FIELD = "temperature";
    inline const static std::string VELOCITY_FIELD = "velocity";

   private:
    const std::shared_ptr<eos::EOS> eos;
    const std::vector<std::string> extraVariables;
    const std::shared_ptr<domain::Region> region;
   public:
    CompressibleFlowFields(std::shared_ptr<eos::EOS>, std::vector<std::string> = {}, std::shared_ptr<domain::Region> = {});

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;

};

}

#endif  // ABLATELIBRARY_COMPRESSIBLEFLOWFIELDS_HPP
