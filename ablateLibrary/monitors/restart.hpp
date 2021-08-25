#ifndef ABLATELIBRARY_RESTART_HPP
#define ABLATELIBRARY_RESTART_HPP

#include "monitor.hpp"

namespace ablate::monitors {
class Restart : public Monitor {
   private:
    const int interval;
    static PetscErrorCode OutputRestart(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx);

   public:
    explicit Restart(int interval = {});

    void Register(std::shared_ptr<Monitorable>) override {}
    PetscMonitorFunction GetPetscFunction() override { return OutputRestart; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_RESTART_HPP
