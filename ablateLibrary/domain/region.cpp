#include "region.hpp"
#include <algorithm>
#include <functional>

ablate::domain::Region::Region(std::string name, std::vector<int> valuesIn) : name(name), values(valuesIn.empty() ? valuesIn : std::vector<int>{1}) {
    // sort the values
    std::sort(values.begin(), values.end());

    // Create a unique string
    auto hashString = name;
    for (const auto& value : values) {
        hashString += "," + std::to_string(value);
    }

    id = std::hash<std::string>()(hashString);
}

#include "parser/registrar.hpp"
REGISTERDEFAULT(ablate::domain::Region, ablate::domain::Region, "The region in which this solver applies (Label & Values)", ARG(std::string, "name", "the label name"),
                OPT(std::vector<int>, "values", "the values on the label (default is 1)"));
