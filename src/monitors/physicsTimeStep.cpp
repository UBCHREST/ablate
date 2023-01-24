#include "physicsTimeStep.hpp"
#include "io/interval/fixedInterval.hpp"
#include "monitors/logs/stdOut.hpp"

ablate::monitors::PhysicsTimeStep::PhysicsTimeStep(std::shared_ptr<logs::Log> logIn, std::shared_ptr<io::interval::Interval> intervalIn)
    : log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

void ablate::monitors::PhysicsTimeStep::Register(std::shared_ptr<solver::Solver> solverIn) {
    Monitor::Register(solverIn);

    solver = std::dynamic_pointer_cast<ablate::solver::PhysicsTimeStepFunction>(solverIn);
    if (!solverIn) {
        throw std::invalid_argument("The PhysicsTimeStep assumes a FiniteVolumeSolver");
    }

    // if this is the first time step init the log
    if (!log->Initialized()) {
        log->Initialize(solverIn->GetSubDomain().GetComm());
    }
}

PetscErrorCode ablate::monitors::PhysicsTimeStep::ReportPhysicsTimeStep(TS ts, PetscInt step, PetscReal crtime, Vec, void* ctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::PhysicsTimeStep*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        auto dts = monitor->solver->ComputePhysicsTimeSteps(ts);

        for (const auto& [name, dt] : dts) {
            monitor->log->Printf("Physics dt %s: %g\n", name.c_str(), dt);
        }
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::PhysicsTimeStep, "Reports the physics based time stepping from the FVM without enforcing it",
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"), OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"));
