#include "compressibleFlowFields.hpp"
#include "domain/fieldDescription.hpp"

ablate::finiteVolume::CompressibleFlowFields::CompressibleFlowFields(std::shared_ptr<eos::EOS> eos, std::vector<std::string> extraVariables, std::shared_ptr<domain::Region> region,
                                                                     std::shared_ptr<parameters::Parameters> conservedFieldParameters)
    : eos(eos), extraVariables(extraVariables), region(region), conservedFieldOptions(conservedFieldParameters) {}

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

    if (!eos->GetSpecies().empty()) {
        flowFields.emplace_back(
            std::make_shared<domain::FieldDescription>(DENSITY_YI_FIELD, DENSITY_YI_FIELD, eos->GetSpecies(), domain::FieldLocation::SOL, domain::FieldType::FVM, region, conservedFieldOptions));
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(YI_FIELD, YI_FIELD, eos->GetSpecies(), domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions));
    }

    if (!extraVariables.empty()) {
        flowFields.emplace_back(
            std::make_shared<domain::FieldDescription>(DENSITY_EV_FIELD, DENSITY_EV_FIELD, extraVariables, domain::FieldLocation::SOL, domain::FieldType::FVM, region, conservedFieldOptions));
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(EV_FIELD, EV_FIELD, extraVariables, domain::FieldLocation::AUX, domain::FieldType::FVM, region, auxFieldOptions));
    }

    return flowFields;
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::CompressibleFlowFields, "FVM fields need for compressible flow",
         ARG(ablate::eos::EOS, "eos", "the equation of state to be used for the flow"), OPT(std::vector<std::string>, "extraVariables", "Any extra variables to transport"),
         OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"),
         OPT(ablate::parameters::Parameters, "conservedFieldOptions", "petsc options used for the conserved fields.  Common options would be petscfv_type and petsclimiter_type"));
