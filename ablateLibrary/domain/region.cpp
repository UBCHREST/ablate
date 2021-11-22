#include "region.hpp"
#include <functional>

ablate::domain::Region::Region(std::string name, int valueIn) : name(name), value(valueIn == 0 ? 1 : valueIn) {
    // Create a unique string
    auto hashString = name + ":" + std::to_string(value);
    id = std::hash<std::string>()(hashString);
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::domain::Region, ablate::domain::Region, "The region in which this solver applies (Label & Values)", ARG(std::string, "name", "the label name"),
                 OPT(int, "value", "the value on the label (default is 1)"));
