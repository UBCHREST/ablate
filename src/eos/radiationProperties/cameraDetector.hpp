#ifndef ABLATELIBRARY_CAMERADETECTOR_HPP
#define ABLATELIBRARY_CAMERADETECTOR_HPP

#include "radiationProperties.hpp"

namespace ablate::eos::radiationProperties {
class CameraDetector : public RadiationModel {

   private:
    const std::shared_ptr<eos::EOS> eos;  //! eos is needed to compute field values

   public:
    CameraDetector(std::shared_ptr<EOS> eosIn);

    ThermodynamicFunction GetRadiationPropertiesFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;
    ThermodynamicTemperatureFunction GetRadiationPropertiesTemperatureFunction(RadiationProperty property, const std::vector<domain::Field>& fields) const;

    static PetscErrorCode CameraFunction(const PetscReal* conserved, PetscReal* kappa, void* ctx);
    static PetscErrorCode CameraTemperatureFunction(const PetscReal* conserved, PetscReal temperature, PetscReal* kappa, void* ctx);

};
}

#endif  // ABLATELIBRARY_CAMERADETECTOR_HPP
