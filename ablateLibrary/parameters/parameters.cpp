#include "parameters.hpp"
#include <set>

static std::set<std::string> knownTrueValues = {"true", "TRUE", "True", "y", "Y", "Yes", "on", "ON", "On", "1", "yes", "YES"};

void ablate::parameters::Parameters::toValue(const std::string& inputString, bool& outputValue) { outputValue = knownTrueValues.count(inputString) > 0; }
