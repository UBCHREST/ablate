#include "compressibleFlowFields.hpp"
#include "domain/fieldDescription.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::finiteVolume::CompressibleFlowFields::CompressibleFlowFields(std::shared_ptr<eos::EOS> eos, std::shared_ptr<domain::Region> region,
                                                                     std::shared_ptr<parameters::Parameters> conservedFieldParameters)
    : eos(eos), region(region), conservedFieldOptions(conservedFieldParameters) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteVolume::CompressibleFlowFields::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> flowFields{
        std::make_shared<domain::FieldDescription>(EULER_FIELD,
                                                   EULER_FIELD,
                                                   std::vector<std::string>{"rho", "rhoE", "rhoVel" + domain::FieldDescription::DIMENSION},
                                                   domain::FieldLocation::SOL,
                                                   domain::FieldType::FVM,
                                                   region,
                                                   conservedFieldOptions),
        std::make_shared<domain::FieldDescription>(
            TEMPERATURE_FIELD, TEMPERATURE_FIELD, domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions),
        std::make_shared<domain::FieldDescription>(
            VELOCITY_FIELD, VELOCITY_FIELD, std::vector<std::string>{"vel" + domain::FieldDescription::DIMENSION}, domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions)};

    if (!eos->GetSpeciesVariables().empty()) {
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(
            DENSITY_YI_FIELD, DENSITY_YI_FIELD, eos->GetSpeciesVariables(), domain::FieldLocation::SOL, domain::FieldType::FVM, region, conservedFieldOptions));
        flowFields.emplace_back(
            std::make_shared<domain::FieldDescription>(YI_FIELD, YI_FIELD, eos->GetSpeciesVariables(), domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions));
    }

    if (!eos->GetProgressVariables().empty()) {
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(DENSITY_PROGRESS_FIELD,
                                                                           DENSITY_PROGRESS_FIELD,
                                                                           eos->GetProgressVariables(),
                                                                           domain::FieldLocation::SOL,
                                                                           domain::FieldType::FVM,
                                                                           region,
                                                                           conservedFieldOptions,
                                                                           std::vector<std::string>{EV_TAG}));
        flowFields.emplace_back(
            std::make_shared<domain::FieldDescription>(PROGRESS_FIELD, PROGRESS_FIELD, eos->GetProgressVariables(), domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions));
    }

    return flowFields;
}

std::istream& ablate::finiteVolume::operator>>(std::istream& is, ablate::finiteVolume::CompressibleFlowFields::ValidRange& v) {
    std::string enumString;
    is >> enumString;
    v = ablate::finiteVolume::CompressibleFlowFields::ValidRange::POSITIVE;
    if (enumString == ablate::finiteVolume::CompressibleFlowFields::BoundRange) {
        v = ablate::finiteVolume::CompressibleFlowFields::ValidRange::BOUND;
    } else if (enumString == ablate::finiteVolume::CompressibleFlowFields::PositiveRange) {
        v = ablate::finiteVolume::CompressibleFlowFields::ValidRange::POSITIVE;
    } else if (enumString == ablate::finiteVolume::CompressibleFlowFields::FullRange) {
        v = ablate::finiteVolume::CompressibleFlowFields::ValidRange::FULL;
    }
    return is;
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::CompressibleFlowFields, "FVM fields need for compressible flow",
         ARG(ablate::eos::EOS, "eos", "the equation of state to be used for the flow"), OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"),
         OPT(ablate::parameters::Parameters, "conservedFieldOptions", "petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type"));
