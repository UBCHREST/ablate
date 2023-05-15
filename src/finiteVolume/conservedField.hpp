#ifndef ABLATELIBRARY_CONSERVEDFIELD_HPP
#define ABLATELIBRARY_CONSERVEDFIELD_HPP

#include <domain/region.hpp>
#include <memory>
#include <string>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "eos/eos.hpp"
#include "parameters/mapParameters.hpp"

namespace ablate::finiteVolume {
/**
 * Helper class to create a conserved and aux field
 */
class ConservedField : public domain::FieldDescriptor {
   protected:
    const std::string name;
    const std::vector<std::string> components;
    const std::shared_ptr<domain::Region> region;
    const bool bound;
    const std::shared_ptr<parameters::Parameters> conservedFieldOptions;
    const std::shared_ptr<parameters::Parameters> auxFieldOptions = ablate::parameters::MapParameters::Create({{"petscfv_type", "leastsquares"}, {"petsclimiter_type", "none"}});

   public:
    /**
     *
     * @param name
     * @param components
     * @param bound , bound the result between zero and one
     * @param conservedFieldParameters
     */
    explicit ConservedField(std::string name, std::vector<std::string> components = {}, std::shared_ptr<domain::Region> = {}, bool bound = false,
                            std::shared_ptr<parameters::Parameters> conservedFieldParameters = {});

    std::vector<std::shared_ptr<domain::FieldDescription>> GetFields() override;
};

}  // namespace ablate::finiteVolume

#endif  // ABLATELIBRARY_CONSERVEDFIELD_HPP
