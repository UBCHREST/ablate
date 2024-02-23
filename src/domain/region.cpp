#include "region.hpp"
#include <functional>
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::Region::Region(const std::string& name, int valueIn) : name(name), value(valueIn == 0 ? 1 : valueIn) {
    // Create a unique string
    auto hashString = name + ":" + std::to_string(value);
    id = std::hash<std::string>()(hashString);
}

void ablate::domain::Region::CreateLabel(DM dm, DMLabel& regionLabel, PetscInt& regionValue) const {
    DMCreateLabel(dm, GetName().c_str()) >> utilities::PetscUtilities::checkError;
    DMGetLabel(dm, GetName().c_str(), &regionLabel) >> utilities::PetscUtilities::checkError;
    regionValue = GetValue();
}

void ablate::domain::Region::GetLabel(const std::shared_ptr<Region>& region, DM dm, DMLabel& regionLabel, PetscInt& regionValue) {
    regionLabel = nullptr;
    regionValue = PETSC_DECIDE;
    if (region) {
        DMGetLabel(dm, region->GetName().c_str(), &regionLabel) >> utilities::PetscUtilities::checkError;
        regionValue = region->GetValue();
    }
}

void ablate::domain::Region::GetLabel(DM dm, DMLabel& regionLabel, PetscInt& regionValue) {
    DMGetLabel(dm, GetName().c_str(), &regionLabel) >> utilities::PetscUtilities::checkError;
    regionValue = GetValue();
}

void ablate::domain::Region::CheckForLabel(DM dm) const {
    DMLabel label = nullptr;
    DMGetLabel(dm, GetName().c_str(), &label) >> utilities::PetscUtilities::checkError;
    if (label == nullptr) {
        throw std::invalid_argument("Unable to locate " + GetName() + " in domain");
    }
}

;
void ablate::domain::Region::CheckForLabel(DM dm, MPI_Comm comm) const {
    DMLabel label = nullptr;
    DMGetLabel(dm, GetName().c_str(), &label) >> utilities::PetscUtilities::checkError;

    auto found = (PetscMPIInt)(label != nullptr);
    PetscMPIInt anyFound;

    MPI_Allreduce(&found, &anyFound, 1, MPI_INT, MPI_MAX, comm) >> utilities::MpiUtilities::checkError;

    if (!anyFound) {
        throw std::invalid_argument("Unable to locate " + GetName() + " in domain");
    }
}

bool ablate::domain::Region::InRegion(const std::shared_ptr<Region>& region, DM dm, PetscInt point) {
    if (!region) {
        return true;
    }
    return region->InRegion(dm, point);
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
