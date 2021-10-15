#include "field.hpp"
#include <map>


static const std::map<std::string, ablate::domain::FieldType> stringToFieldLocation = {
    {"sol", ablate::domain::FieldType::SOL}, {"SOL", ablate::domain::FieldType::SOL}, {"AUX", ablate::domain::FieldType::AUX}, {"aux", ablate::domain::FieldType::AUX}};

std::istream& ablate::domain::operator>>(std::istream& is, ablate::domain::FieldType& v) {
    // get the key string
    std::string enumString;
    is >> enumString;
    v = stringToFieldLocation.at(enumString);
    return is;
}