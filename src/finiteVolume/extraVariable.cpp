#include "extraVariable.hpp"

#include <utility>
#include "domain/fieldDescription.hpp"

ablate::finiteVolume::ExtraVariable::ExtraVariable(const std::string& name, std::vector<std::string> components, std::shared_ptr<domain::Region> region, CompressibleFlowFields::ValidRange range,
                                                   std::shared_ptr<parameters::Parameters> conservedFieldParameters)
    : name(name.empty() ? finiteVolume::CompressibleFlowFields::EV_FIELD : name),
      components(std::move(components)),
      region(std::move(region)),
      range(range),
      conservedFieldOptions(std::move(conservedFieldParameters)) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteVolume::ExtraVariable::GetFields() {
    std::vector<std::string> evTags = {CompressibleFlowFields::EV_TAG};

    switch (range) {
        case CompressibleFlowFields::ValidRange::BOUND:
            evTags.push_back(CompressibleFlowFields::BoundRange);
            break;
        case CompressibleFlowFields::ValidRange::POSITIVE:
            evTags.push_back(CompressibleFlowFields::PositiveRange);
            break;
        case CompressibleFlowFields::ValidRange::FULL:
            evTags.push_back(CompressibleFlowFields::FullRange);
            break;
    }

    return {
        std::make_shared<domain::FieldDescription>(
            CompressibleFlowFields::CONSERVED + name, CompressibleFlowFields::CONSERVED + name, components, domain::FieldLocation::SOL, domain::FieldType::FVM, region, conservedFieldOptions, evTags),
        std::make_shared<domain::FieldDescription>(name, name, components, domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions)};
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::ExtraVariable, "Helper function to setup a conserved flow field", OPT(std::string, "name", "the name of the field, default is EV"),
         OPT(std::vector<std::string>, "components", "Optional field components"), OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"),
         OPT(EnumWrapper<ablate::finiteVolume::CompressibleFlowFields::ValidRange>, "range", "valid range of the extra variable; full (default), positive, bound (0 to 1)"),
         OPT(ablate::parameters::Parameters, "conservedFieldOptions", "petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type"));
