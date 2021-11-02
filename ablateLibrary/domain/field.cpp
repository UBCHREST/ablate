#include "field.hpp"
#include <map>

static const std::map<std::string, ablate::domain::FieldType> stringToFieldLocation = {
    {"sol", ablate::domain::FieldType::SOL}, {"SOL", ablate::domain::FieldType::SOL}, {"AUX", ablate::domain::FieldType::AUX}, {"aux", ablate::domain::FieldType::AUX}};

static const std::map<std::string, ablate::domain::FieldAdjacency> stringToFieldAdjacency = {{"default", ablate::domain::FieldAdjacency::DEFAULT},
                                                                                             {"DEFAULT", ablate::domain::FieldAdjacency::DEFAULT},
                                                                                             {"FEM", ablate::domain::FieldAdjacency::FEM},
                                                                                             {"fem", ablate::domain::FieldAdjacency::FEM},
                                                                                             {"FVM", ablate::domain::FieldAdjacency::FVM},
                                                                                             {"fvm", ablate::domain::FieldAdjacency::FVM}};

std::istream& ablate::domain::operator>>(std::istream& is, ablate::domain::FieldType& v) {
    // get the key string
    std::string enumString;
    is >> enumString;
    v = stringToFieldLocation.at(enumString);
    return is;
}

std::istream& ablate::domain::operator>>(std::istream& is, ablate::domain::FieldAdjacency& v) {
    std::string enumString;
    is >> enumString;
    v = stringToFieldAdjacency.at(enumString);
    return is;
}
