#ifndef ABLATELIBRARY_RADIATIONPROPERTIESZIMMER_H
#define ABLATELIBRARY_RADIATIONPROPERTIESZIMMER_H

#include "radiationProperties.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "utilities/mathUtilities.hpp"
#include "solver/cellSolver.hpp"
#include "eos/radiationProperties/radiationProperties.hpp"

namespace ablate::eos::radiationProperties {

class Zimmer : public RadiationModel {

   private:
    struct FunctionContext {
        PetscInt densityYiH2OOffset;
        PetscInt densityYiCO2Offset;
        PetscInt densityYiCOOffset;
        PetscInt densityYiCH4Offset;
    };

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode ZimmerFunction(const PetscReal conserved[], PetscReal* property, void* ctx);

    /**
     * private static function for evaluating constant properties without temperature
     * @param conserved
     * @param property
     * @param ctx
     */
    static PetscErrorCode ZimmerTemperatureFunction(const PetscReal conserved[], PetscReal temperature, PetscReal* property, void* ctx);

   public:
    explicit Zimmer(std::shared_ptr<parameters::Parameters> options);
    explicit Zimmer(const Zimmer&) = delete;
    void operator=(const Zimmer&) = delete;

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

}

#endif  // ABLATELIBRARY_RADIATIONPROPERTIESZIMMER_H
