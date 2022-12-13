#ifndef ABLATELIBRARY_RADIATIONPROPERTIESADD_HPP
#define ABLATELIBRARY_RADIATIONPROPERTIESADD_HPP

#include <map>
#include <memory>
#include "radiationProperties.hpp"

namespace ablate::eos::radiationProperties {

/**
 * Simple helper model that sums the properties for any supplied model
 */
class Sum : public RadiationModel {
   private:
    /**
     * The list of supplied models
     */
    const std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>> models;

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode SumFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode SumTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);

   public:
    explicit Sum(std::vector<std::shared_ptr<ablate::eos::radiationProperties::RadiationModel>> models);
    explicit Sum(const Sum&) = delete;
    void operator=(const Sum&) = delete;

    /**
     * Single function to produce radiation properties function for any property based upon the available fields
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;

    /**
     * Single function to produce thermodynamic function for any property based upon the available fields and temperature
     * @param property
     * @param fields
     * @return
     */
    [[nodiscard]] ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const override;
};

}  // namespace ablate::eos::radiationProperties

#endif  // ABLATELIBRARY_RADIATIONPROPERTIESZIMMER_H
