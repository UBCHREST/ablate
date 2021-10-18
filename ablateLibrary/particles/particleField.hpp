#ifndef ABLATELIBRARY_PARTICLEFIELD_HPP
#define ABLATELIBRARY_PARTICLEFIELD_HPP

#include <petsc.h>
#include <string>
#include <vector>
#include "domain/field.hpp"

namespace ablate::particles {

struct ParticleField {
    const std::string name;
    const std::vector<std::string> components = {"_"};
    enum domain::FieldType type = domain::FieldType::AUX;
    const PetscDataType dataType;
};

}  // namespace ablate::particles
#endif  // ABLATELIBRARY_FIELD_HPP
