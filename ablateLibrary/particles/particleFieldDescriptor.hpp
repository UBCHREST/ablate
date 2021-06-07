#ifndef ABLATELIBRARY_PARTICLEFIELDDESCRIPTOR_HPP
#define ABLATELIBRARY_PARTICLEFIELDDESCRIPTOR_HPP

#include <petsc.h>
namespace ablate::particles {

struct ParticleFieldDescriptor {
    const std::string fieldName;
    const PetscInt components;
    const PetscDataType type;
};
}

#endif  // ABLATELIBRARY_PARTICLEFIELDDESCRIPTOR_HPP
