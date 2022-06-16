#include "levelSetField.hpp"

ablate::levelSet::LevelSetField::LevelSetField(std::shared_ptr<domain::Region> region) : region(region) {}

std::vector<std::shared_ptr<ablate::domain::FieldDescription>> ablate::levelSet::LevelSetField::GetFields() {
    std::vector<std::shared_ptr<ablate::domain::FieldDescription>> levelSetField{
        std::make_shared<domain::FieldDescription>("level set field", "phi", domain::FieldDescription::ONECOMPONENT, domain::FieldLocation::SOL, domain::FieldType::FVM, region)};

    return levelSetField;
}

#include "registrar.hpp"
REGISTER(ablate::domain::FieldDescriptor, ablate::levelSet::LevelSetField, "Level Set fields need for interface tracking",
         OPT(ablate::domain::Region, "region", "the region for the level set (defaults to entire domain)"));
