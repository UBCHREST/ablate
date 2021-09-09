#ifndef ABLATELIBRARY_SERIALIZER_HPP
#define ABLATELIBRARY_SERIALIZER_HPP
#include <petsc.h>
#include "serializable.hpp"

namespace ablate::io {

typedef PetscErrorCode (*PetscSerializeFunction)(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

/**
 * The Register method registers the class to be saved and restores any values if present.
 */
class Serializer {
   public:
    virtual ~Serializer() = default;
    virtual void Register(std::weak_ptr<Serializable>) = 0;
    virtual void* GetContext() { return this; }
    virtual PetscSerializeFunction GetSerializeFunction() = 0;
    virtual void RestoreTS(TS ts) = 0;
};
}  // namespace ablate::io

#endif  // ABLATELIBRARY_SERIALIZER_HPP
