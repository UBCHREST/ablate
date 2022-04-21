#include "modifier.hpp"
std::ostream& ablate::domain::modifiers::operator<<(std::ostream& os, const ablate::domain::modifiers::Modifier& modifier) {
    os << modifier.ToString();
    return os;
}
