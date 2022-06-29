#ifndef ABLATELIBRARY_ROCKETMONITOR_HPP
#define ABLATELIBRARY_ROCKETMONITOR_HPP

#include <petsc.h>
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::monitors {

class RocketMonitor : public Monitor {
   private:
    static PetscErrorCode OutputRocket(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);
    const std::shared_ptr<domain::Region> region;
    const std::shared_ptr<domain::Region> fieldBoundary;
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<io::interval::Interval> interval;

   public:
    RocketMonitor(std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary, std::shared_ptr<logs::Log> log = {}, std::shared_ptr<io::interval::Interval> interval = {});
    ~RocketMonitor() override;

    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputRocket; }
};

}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_ROCKETMONITOR_HPP
