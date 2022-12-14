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

    //! the primary euler field containing the EulerComponents
    inline const static std::string EULER_FIELD = "euler";

    //! The conserved prefix used for fields that have a conserved and non conserved form
    inline const static std::string CONSERVED = "density";

    //! the conserved (density*yi) solution field for species mass fractions
    inline const static std::string YI_FIELD = "Yi";
    inline const static std::string DENSITY_YI_FIELD = CONSERVED + YI_FIELD;

    //! progress fields are used by the eos/chemistry model to transport required non species
    inline const static std::string PROGRESS_FIELD = "Progress";
    inline const static std::string DENSITY_PROGRESS_FIELD = CONSERVED + PROGRESS_FIELD;

    //! the conserved tag used to tag all fields that should act like extra variables (transported with the flow)
    inline const static std::string EV_TAG = "EV";

    //! these are arbitrary ev fields
    inline const static std::string EV_FIELD = "EV";
    inline const static std::string DENSITY_EV_FIELD = CONSERVED + EV_FIELD;

    //! some common aux fields
    inline const static std::string TEMPERATURE_FIELD = "temperature";
    inline const static std::string VELOCITY_FIELD = "velocity";
    inline const static std::string PRESSURE_FIELD = "pressure";

   protected:
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
