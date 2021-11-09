#include "compressibleFlowFields.hpp"
#include "domain/fieldDescription.hpp"

ablate::finiteVolume::CompressibleFlowFields::CompressibleFlowFields(std::shared_ptr<eos::EOS> eos, std::vector<std::string> extraVariables, std::shared_ptr<domain::Region> region)
    : eos(eos), extraVariables(extraVariables), region(region) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteVolume::CompressibleFlowFields::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> flowFields{
        std::make_shared<domain::FieldDescription>(
            EULER_FIELD, EULER_FIELD, std::vector<std::string>{"rho", "rhoE", "rhoVel" + domain::FieldDescription::DIMENSION}, domain::FieldLocation::SOL, domain::FieldType::FVM, region),
        std::make_shared<domain::FieldDescription>(TEMPERATURE_FIELD, TEMPERATURE_FIELD, domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::AUX, domain::FieldType::FVM, region),
        std::make_shared<domain::FieldDescription>(
            VELOCITY_FIELD, VELOCITY_FIELD, std::vector<std::string>{"vel" + domain::FieldDescription::DIMENSION}, domain::FieldLocation::AUX, domain::FieldType::FVM, region)};

    if (!eos->GetSpecies().empty()) {
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(DENSITY_YI_FIELD, DENSITY_YI_FIELD, eos->GetSpecies(), domain::FieldLocation::SOL, domain::FieldType::FVM, region));
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(YI_FIELD, YI_FIELD, eos->GetSpecies(), domain::FieldLocation::AUX, domain::FieldType::FVM, region));
    }

    if (!extraVariables.empty()) {
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(DENSITY_EV_FIELD, DENSITY_EV_FIELD, extraVariables, domain::FieldLocation::SOL, domain::FieldType::FVM, region));
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(EV_FIELD, EV_FIELD, extraVariables, domain::FieldLocation::AUX, domain::FieldType::FVM, region));
    }

    return flowFields;
}

#include "parser/registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteVolume::CompressibleFlowFields, "FVM fields need for compressible flow", ARG(eos::EOS, "eos", "the equation of state to be used for the flow"),
         OPT(std::vector<std::string>, "extraVariables", "Any extra variables to transport"), OPT(domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"));
