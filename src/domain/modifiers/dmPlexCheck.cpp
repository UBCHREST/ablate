#include "dmPlexCheck.hpp"
#include <utilities/petscError.hpp>

void ablate::domain::modifiers::DMPlexCheck::Modify(DM &dm) { ::DMPlexCheck(dm) >> checkError; }
std::string ablate::domain::modifiers::DMPlexCheck::ToString() const { return "ablate::domain::modifiers::DMPlexCheck"; }

#include "registrar.hpp"
REGISTER_WITHOUT_ARGUMENTS(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::DMPlexCheck,
                           "Calls the [DMPlexCheck](https://petsc.org/main/docs/manualpages/DMPLEX/DMPlexCheck/) petsc function");