#include "field.hpp"
#include <map>
#include "fieldDescription.hpp"

static const std::map<std::string, ablate::domain::FieldLocation> stringToFieldLocation = {{"", ablate::domain::FieldLocation::SOL},
                                                                                           {"sol", ablate::domain::FieldLocation::SOL},
                                                                                           {"SOL", ablate::domain::FieldLocation::SOL},
                                                                                           {"AUX", ablate::domain::FieldLocation::AUX},
                                                                                           {"aux", ablate::domain::FieldLocation::AUX}};

static const std::map<std::string, ablate::domain::FieldType> stringToFieldAdjacency = {{"FE", ablate::domain::FieldType::FEM},
                                                                                        {"fe", ablate::domain::FieldType::FEM},
                                                                                        {"FEM", ablate::domain::FieldType::FEM},
                                                                                        {"fem", ablate::domain::FieldType::FEM},
                                                                                        {"FVM", ablate::domain::FieldType::FVM},
                                                                                        {"fvm", ablate::domain::FieldType::FVM},
                                                                                        {"FV", ablate::domain::FieldType::FVM},
                                                                                        {"fv", ablate::domain::FieldType::FVM}};

std::istream& ablate::domain::operator>>(std::istream& is, ablate::domain::FieldLocation& v) {
    // get the key string
    std::string enumString;
    is >> enumString;
    v = stringToFieldLocation.at(enumString);
    return is;
}

std::istream& ablate::domain::operator>>(std::istream& is, ablate::domain::FieldType& v) {
    std::string enumString;
    is >> enumString;
    v = stringToFieldAdjacency.at(enumString);
    return is;
}

ablate::domain::Field ablate::domain::Field::FromFieldDescription(const ablate::domain::FieldDescription& fieldDescription, PetscInt id, PetscInt subId, PetscInt offset) {
    return ablate::domain::Field{.name = fieldDescription.name,
                                 .numberComponents = (PetscInt)fieldDescription.components.size(),
                                 .components = fieldDescription.components,
                                 .id = id,
                                 .subId = subId,
                                 .offset = offset,
                                 .location = fieldDescription.location,
                                 .type = fieldDescription.type,
                                 .tags = std::set<std::string>(fieldDescription.tags.begin(), fieldDescription.tags.end())};
}

ablate::domain::Field ablate::domain::Field::CreateSubField(PetscInt newSubId, PetscInt newOffset) const {
    return ablate::domain::Field{
        .name = name, .numberComponents = numberComponents, .components = components, .id = id, .subId = newSubId, .offset = newOffset, .location = location, .type = type, .tags = tags};
}
