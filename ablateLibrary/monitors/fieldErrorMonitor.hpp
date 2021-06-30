#ifndef ABLATELIBRARY_FIELDERRORMONITOR_HPP
#define ABLATELIBRARY_FIELDERRORMONITOR_HPP
#include "monitor.hpp"
namespace ablate::monitors {

class FieldErrorMonitor : public Monitor {
   private:
    static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx);

   public:
    void Register(std::shared_ptr<Monitorable>) override {}
    PetscMonitorFunction GetPetscFunction() override { return MonitorError; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_FIELDERRORMONITOR_HPP