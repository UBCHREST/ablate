#include "region.hpp"
#include <functional>
#include "utilities/petscError.hpp"

ablate::domain::Region::Region(std::string name, int valueIn) : name(name), value(valueIn == 0 ? 1 : valueIn) {
    // Create a unique string
    auto hashString = name + ":" + std::to_string(value);
    id = std::hash<std::string>()(hashString);
}

void ablate::domain::Region::CreateLabel(DM dm, DMLabel& regionLabel, PetscInt& regionValue) {
    DMCreateLabel(dm, GetName().c_str()) >> checkError;
    DMGetLabel(dm, GetName().c_str(), &regionLabel) >> checkError;
    regionValue = GetValue();
}

void ablate::domain::Region::GetLabel(const std::shared_ptr<Region>& region, DM dm, DMLabel& regionLabel, PetscInt& regionValue) {
    regionLabel = nullptr;
    regionValue = PETSC_DECIDE;
    if (region) {
        DMGetLabel(dm, region->GetName().c_str(), &regionLabel) >> checkError;
        regionValue = region->GetValue();
    }
}
void ablate::domain::Region::CheckForLabel(DM dm) const {
    DMLabel label = nullptr;
    DMGetLabel(dm, GetName().c_str(), &label) >> checkError;
    if (label == nullptr) {
        throw std::invalid_argument("Unable to locate " + GetName() + " in domain");
    }
}

bool ablate::domain::Region::InRegion(const std::shared_ptr<Region>& region, DM dm, PetscInt point) {
    if (!region) {
        return true;
    }
    PetscInt ptValue;
    DMGetLabelValue(dm, region->name.c_str(), point, &ptValue) >> checkError;
    return ptValue == region->value;
}

std::ostream& ablate::domain::operator<<(std::ostream& os, const ablate::domain::Region& region) {
    os << region.ToString();
    return os;
}

std::ostream& ablate::domain::operator<<(std::ostream& os, const std::shared_ptr<ablate::domain::Region>& region) {
    if (region) {
        os << *region;
    } else {
        os << "EntireDomain";
    }
    return os;
}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::domain::Region, ablate::domain::Region, "The region in which this solver applies (Label & Values)", ARG(std::string, "name", "the label name"),
                 OPT(int, "value", "the value on the label (default is 1)"));
