#ifndef ABLATELIBRARY_ROCKETMONITOR_HPP
#define ABLATELIBRARY_ROCKETMONITOR_HPP

#include <petsc.h>
#include "domain/region.hpp"
#include "domain/subDomain.hpp"
#include "eos/eos.hpp"
#include "finiteVolume/boundaryConditions/ghost.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "io/interval/interval.hpp"
#include "monitor.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::monitors {

class RocketMonitor : public Monitor {
   private:
    static PetscErrorCode OutputRocket(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx);
    const std::string name;
    const std::shared_ptr<domain::Region> region;
    const std::shared_ptr<domain::Region> fieldBoundary;
    const std::shared_ptr<eos::EOS> eos;
    eos::ThermodynamicFunction computePressure;
    eos::ThermodynamicFunction computeSpeedOfSound;
    const std::shared_ptr<logs::Log> log;
    const std::shared_ptr<io::interval::Interval> interval;
    double referencePressure;

   public:
    RocketMonitor(const std::string name, std::shared_ptr<domain::Region> region, std::shared_ptr<domain::Region> fieldBoundary, std::shared_ptr<eos::EOS> eos,
                  const std::shared_ptr<logs::Log>& log = {}, const std::shared_ptr<io::interval::Interval>& interval = {}, double referencePressure = {});
    void Register(std::shared_ptr<solver::Solver>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputRocket; }
};

}  // namespace ablate::monitors
#endif  // ABLATELIBRARY_ROCKETMONITOR_HPP
