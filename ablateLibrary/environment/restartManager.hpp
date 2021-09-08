#ifndef ABLATELIBRARY_RESTARTMANAGER_HPP
#define ABLATELIBRARY_RESTARTMANAGER_HPP
#include <memory>
#include "restartable.hpp"
#include <filesystem>

namespace ablate::environment {

typedef PetscErrorCode (*PetscSaveStateFunction)(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

class RestartManager {
   public:
    /**
     * The Register method registers the class to be saved and restores any values if present.
     */
    virtual ~RestartManager() = default;
    virtual void Register(std::weak_ptr<Restartable>) = 0;
    virtual void* GetContext() { return this; }
    virtual PetscSaveStateFunction GetTSFunction() = 0;
    virtual std::filesystem::path GetInputPath() const= 0;
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_RESTARTMANAGER_HPP
