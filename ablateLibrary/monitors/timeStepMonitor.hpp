#ifndef ABLATELIBRARY_TIMESTEPMONITOR_HPP
#define ABLATELIBRARY_TIMESTEPMONITOR_HPP
#include <memory>
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::monitors {
class TimeStepMonitor : public Monitor {
   private:
    static PetscErrorCode MonitorTimeStep(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<io::interval::Interval> interval;

   public:
    explicit TimeStepMonitor(std::shared_ptr<logs::Log> log = {}, std::shared_ptr<io::interval::Interval> interval = {});

    PetscMonitorFunction GetPetscFunction() override { return MonitorTimeStep; }
};
}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_TIMESTEPMONITOR_HPP
