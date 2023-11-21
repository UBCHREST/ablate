#include "createCoordinateSpace.hpp"
#include "utilities/petscUtilities.hpp"

ablate::domain::modifiers::CreateCoordinateSpace::CreateCoordinateSpace(int degree) : degree(degree) {}

void ablate::domain::modifiers::CreateCoordinateSpace::Modify(DM &dm) { DMPlexCreateCoordinateSpace(dm, degree, PETSC_TRUE, nullptr) >> utilities::PetscUtilities::checkError; }
std::string ablate::domain::modifiers::CreateCoordinateSpace::ToString() const { return "ablate::domain::modifiers::CreateCoordinateSpace"; }

#include "registrar.hpp"
REGISTER(ablate::domain::modifiers::Modifier, ablate::domain::modifiers::CreateCoordinateSpace,
         "Wrapper for [DMPlexCreateCoordinateSpace](https://petsc.org/release/docs/manualpages/DMPLEX/DMPlexLabelComplete.html)", ARG(int, "degree", "degree of the finite element"));