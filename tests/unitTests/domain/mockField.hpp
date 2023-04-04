#ifndef ABLATELIBRARY_MOCKFIELD_HPP
#define ABLATELIBRARY_MOCKFIELD_HPP

#include <utility>

#include "domain/field.hpp"

namespace ablateTesting::domain::MockField {
inline ablate::domain::Field Create(std::string name, PetscInt numberComponents = 1, PetscInt offset = 0, std::set<std::string> tags = {}) {
    return ablate::domain::Field{.name = std::move(name),
                                 .numberComponents = numberComponents,
                                 .components = {},
                                 .id = -1,
                                 .subId = PETSC_DEFAULT,
                                 .offset = offset,
                                 .location = ablate::domain::FieldLocation::SOL,
                                 .type = ablate::domain::FieldType::FVM,
                                 .tags = tags};
}
inline ablate::domain::Field Create(std::string name, const std::vector<std::string>& components, PetscInt offset = 0, std::set<std::string> tags = {}) {
    return ablate::domain::Field{.name = std::move(name),
                                 .numberComponents = (PetscInt)components.size(),
                                 .components = components,
                                 .id = -1,
                                 .subId = PETSC_DEFAULT,
                                 .offset = offset,
                                 .location = ablate::domain::FieldLocation::SOL,
                                 .type = ablate::domain::FieldType::FVM,
                                 .tags = tags};
}

}  // namespace ablateTesting::domain::MockField

#endif  // ABLATELIBRARY_MOCKFIELD_HPP
