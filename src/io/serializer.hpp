#ifndef ABLATELIBRARY_SERIALIZER_HPP
#define ABLATELIBRARY_SERIALIZER_HPP
#include <petsc.h>
#include <memory>
#include "serializable.hpp"
#include "utilities/petscUtilities.hpp"

namespace ablate::io {

typedef PetscErrorCode (*PetscSerializeFunction)(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

/**
 * The Register method registers the class to be saved and restores any values if present.
 */
class Serializer {
   public:
    /**
     * Allow Serializer to cleanup
     */
    virtual ~Serializer() = default;

    /**
     * Register any solver, process, or class to serialize
     */
    virtual void Register(std::weak_ptr<Serializable>) = 0;

    /**
     * Provide petsc style context to the TS
     * @return
     */
    virtual void* GetContext() { return this; }

    /**
     * Provide petsc style function to the TS
     * @return
     */
    virtual PetscSerializeFunction GetSerializeFunction() = 0;

    /**
     * Restore ts to last saved state
     * @param ts
     */
    virtual void RestoreTS(TS ts) = 0;

    /**
     * Manually call the save for this seralizer
     */
    PetscErrorCode Serialize(TS ts, PetscInt steps, PetscReal time, Vec u) {
        PetscFunctionBeginUser;
        PetscCall(GetSerializeFunction()(ts, steps, time, u, GetContext()));
        PetscFunctionReturn(0);
    }
};
}  // namespace ablate::io

#endif  // ABLATELIBRARY_SERIALIZER_HPP
