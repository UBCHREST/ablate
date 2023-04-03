#ifndef ABLATELIBRARY_MOCKFIELD_HPP
#define ABLATELIBRARY_MOCKFIELD_HPP

#include "domain/field.hpp"

namespace ablateTesting::domain::MockField {
ablate::domain::Field Create(std::string name, PetscInt numberComponents = 1, PetscInt offset = 0) {
    return ablate::domain::Field{.name = name,
                                 .numberComponents = numberComponents,
                                 .components = {},
                                 .id = -1,
                                 .subId = PETSC_DEFAULT,
                                 .offset = offset,
                                 .location = ablate::domain::FieldLocation::SOL,
                                 .type = ablate::domain::FieldType::FVM,
                                 .tags = {}};
}
}  // namespace ablateTesting::domain::MockField

#endif  // ABLATELIBRARY_MOCKFIELD_HPP
