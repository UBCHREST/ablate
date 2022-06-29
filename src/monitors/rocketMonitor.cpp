#include "rocketMonitor.hpp"
#include "io/interval/fixedInterval.hpp"
#include "monitors/logs/stdOut.hpp"
#include "monitors/logs/log.hpp"


ablate::monitors::RocketMonitor::RocketMonitor(std::shared_ptr<domain::Region> regionIn, std::shared_ptr<domain::Region> fieldBoundaryIn,std::shared_ptr<logs::Log> logIn, std::shared_ptr<io::interval::Interval> intervalIn)
    : log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

PetscErrorCode ablate::monitors::RocketMonitor::OutputRocket(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {

}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RocketMonitor, "Outputs the Thrust and Specific Impulse of a Rocket",
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"), OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"));
