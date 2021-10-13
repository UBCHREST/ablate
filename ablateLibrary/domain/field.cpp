#include "field.hpp"
#include <map>


static const std::map<std::string, ablate::domain::FieldLocation> stringToFieldLocation = {
    {"sol", ablate::domain::FieldLocation::SOL}, {"SOL", ablate::domain::FieldLocation::SOL}, {"AUX", ablate::domain::FieldLocation::AUX}, {"aux", ablate::domain::FieldLocation::AUX}};

std::istream& ablate::domain::operator>>(std::istream& is, ablate::domain::FieldLocation& v) {
    // get the key string
    std::string enumString;
    is >> enumString;
    v = stringToFieldLocation.at(enumString);
    return is;
}