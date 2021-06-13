#ifndef ABLATELIBRARY_ERRORMONITOR_HPP
#define ABLATELIBRARY_ERRORMONITOR_HPP
#include "monitor.hpp"
namespace ablate::monitors {

class ErrorMonitor : public Monitor {
   private:
    static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);

   public:
    void Register(std::shared_ptr<Monitorable>) override {}
    PetscMonitorFunction GetPetscFunction() override { return MonitorError; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_ERRORMONITOR_HPP
