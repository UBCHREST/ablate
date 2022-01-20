#ifndef ABLATELIBRARY_PHYSICSTIMESTEP_HPP
#define ABLATELIBRARY_PHYSICSTIMESTEP_HPP

#include <memory>
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::monitors {

/**
 * Reports the physics based time stepping from the FVM without enforcing it
 */
class PhysicsTimeStep : public Monitor {
   private:
    static PetscErrorCode ReportPhysicsTimeStep(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<io::interval::Interval> interval;
    std::shared_ptr<ablate::finiteVolume::FiniteVolumeSolver> fvmSolver;

   public:
    explicit PhysicsTimeStep(std::shared_ptr<logs::Log> log = {}, std::shared_ptr<io::interval::Interval> interval = {});

    void Register(std::shared_ptr<solver::Solver>) override;

    PetscMonitorFunction GetPetscFunction() override { return ReportPhysicsTimeStep; }
};

}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_PHYSICSTIMESTEP_HPP
