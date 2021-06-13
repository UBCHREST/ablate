#ifndef ABLATELIBRARY_TIMESTEPMONITOR_HPP
#define ABLATELIBRARY_TIMESTEPMONITOR_HPP
#include "monitor.hpp"

namespace ablate::monitors {
class TimeStepMonitor : public Monitor {
   private:
    static PetscErrorCode MonitorTimeStep(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);

   public:
    void Register(std::shared_ptr<Monitorable>) override {}
    PetscMonitorFunction GetPetscFunction() override { return MonitorTimeStep; }
};
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_TIMESTEPMONITOR_HPP
