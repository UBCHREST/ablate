#include "turbulenceFlowFields.hpp"
#include "domain/fieldDescription.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::TurbulenceFlowFields::TurbulenceFlowFields(std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> conservedFieldParameters)
    : region(region), conservedFieldOptions(conservedFieldParameters) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteVolume::TurbulenceFlowFields::GetFields() {
    return {std::make_shared<domain::FieldDescription>(DENSITY_TKE_FIELD,
                                                       DENSITY_TKE_FIELD,
                                                       domain::FieldDescription::ONECOMPONENT,
                                                       domain::FieldLocation::SOL,
                                                       domain::FieldType::FVM,
                                                       region,
                                                       conservedFieldOptions,
                                                       std::vector<std::string>{CompressibleFlowFields::EV_TAG}),
            std::make_shared<domain::FieldDescription>(TKE_FIELD, TKE_FIELD, domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions)};
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::TurbulenceFlowFields, "FVM fields need for the turbulence fields",
         OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"),
         OPT(ablate::parameters::Parameters, "conservedFieldOptions", "petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type"));
