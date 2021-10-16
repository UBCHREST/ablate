#ifndef ABLATELIBRARY_FIELDERRORMONITOR_HPP
#define ABLATELIBRARY_FIELDERRORMONITOR_HPP
#include <monitors/logs/log.hpp>
#include "monitor.hpp"
namespace ablate::monitors {

class FieldErrorMonitor : public Monitor {
   private:
    static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);
    const std::shared_ptr<logs::Log> log;

   public:
    explicit FieldErrorMonitor(std::shared_ptr<logs::Log> log = {});

    void Register(std::shared_ptr<solver::Solver>) override {}
    PetscMonitorFunction GetPetscFunction() override { return MonitorError; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_FIELDERRORMONITOR_HPP