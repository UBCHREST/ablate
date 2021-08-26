#ifndef ABLATELIBRARY_TIMESTEPMONITOR_HPP
#define ABLATELIBRARY_TIMESTEPMONITOR_HPP
#include <monitors/logs/log.hpp>
#include "monitor.hpp"

namespace ablate::monitors {
class TimeStepMonitor : public Monitor {
   private:
    static PetscErrorCode MonitorTimeStep(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);
    const std::shared_ptr<logs::Log> log;
    const int interval;

   public:
    explicit TimeStepMonitor(std::shared_ptr<logs::Log> log = {}, int interval = {});

    void Register(std::shared_ptr<Monitorable>) override {}
    PetscMonitorFunction GetPetscFunction() override { return MonitorTimeStep; }
};
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_TIMESTEPMONITOR_HPP
