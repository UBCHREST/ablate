#include "lowMachFlowFields.hpp"
#include "domain/fieldDescription.hpp"

ablate::finiteElement::LowMachFlowFields::LowMachFlowFields(std::shared_ptr<domain::Region> region, bool includeSourceTerms) : region(region), includeSourceTerms(includeSourceTerms) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::finiteElement::LowMachFlowFields::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> flowFields{
        std::make_shared<domain::FieldDescription>(
            "velocity", "vel", std::vector<std::string>{"vel" + domain::FieldDescription::DIMENSION}, domain::FieldLocation::SOL, domain::FieldType::FEM, region),
        std::make_shared<domain::FieldDescription>("pressure", "pres", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FEM, region),
        std::make_shared<domain::FieldDescription>("temperature", "temp", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FEM, region)};

    if (includeSourceTerms) {
        flowFields.emplace_back(std::make_shared<domain::FieldDescription>(
            "momentum_source", "momentum_source", std::vector<std::string>{"mom" + domain::FieldDescription::DIMENSION}, domain::FieldLocation::AUX, domain::FieldType::FEM, region));
        flowFields.emplace_back(
            std::make_shared<domain::FieldDescription>("mass_source", "mass_source", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::AUX, domain::FieldType::FEM, region));
        flowFields.emplace_back(
            std::make_shared<domain::FieldDescription>("energy_source", "energy_source", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::AUX, domain::FieldType::FEM, region));
    }

    return flowFields;
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::finiteElement::LowMachFlowFields, "FE fields need for incompressible/low-Mach flow",
         OPT(ablate::domain::Region, "region", "the region for the compressible flow (defaults to entire domain)"),
         OPT(bool, "includeSourceTerms", "include aux fields for source terms (defaults to false)"));