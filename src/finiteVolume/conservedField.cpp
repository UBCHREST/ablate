#include "conservedField.hpp"
#include "compressibleFlowFields.hpp"
#include "domain/fieldDescription.hpp"

ablate::finiteVolume::ConservedField::ConservedField(std::string name, std::vector<std::string> components, std::shared_ptr<domain::Region> region,
                                                     std::shared_ptr<parameters::Parameters> conservedFieldParameters)
    : name(name), components(components), region(region), conservedFieldOptions(conservedFieldParameters) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteVolume::ConservedField::GetFields() {
    return {std::make_shared<domain::FieldDescription>(CompressibleFlowFields::CONSERVED + name,
                                                       CompressibleFlowFields::CONSERVED + name,
                                                       components,
                                                       domain::FieldLocation::SOL,
                                                       domain::FieldType::FVM,
                                                       region,
                                                       conservedFieldOptions,
                                                       std::vector<std::string>{CompressibleFlowFields::EV_TAG}),
            std::make_shared<domain::FieldDescription>(name, name, components, domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions)};
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::ConservedField, "Helper function to setup a conserved flow field", ARG(std::string, "name", "the name of the field"),
         OPT(std::vector<std::string>, "components", "Optional field components"), OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"),
         OPT(ablate::parameters::Parameters, "conservedFieldOptions", "petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type"));
