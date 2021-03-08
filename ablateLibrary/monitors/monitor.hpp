#ifndef ABLATELIBRARY_MONITOR_HPP
#define ABLATELIBRARY_MONITOR_HPP
#include <petsc.h>

namespace ablate::monitors{

typedef PetscErrorCode (*PetscMonitorFunction)(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

class Monitor {
   public:
    virtual ~Monitor() = default;
    virtual void* GetContext() {
        return this;
    }

    virtual PetscMonitorFunction GetPetscFunction() = 0;
};

}

#endif  // ABLATELIBRARY_MONITOR_HPP
