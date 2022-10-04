#include "delayInterval.hpp"
#include "utilities/mpiError.hpp"

ablate::io::interval::DelayInterval::DelayInterval(std::shared_ptr<Interval> interval, double minimumSimulationTime, int minimumSimulationStep)
    : interval(interval), minimumSimulationTime(minimumSimulationTime), minimumSimulationStep(minimumSimulationStep) {}

bool ablate::io::interval::DelayInterval::Check(MPI_Comm comm, PetscInt steps, PetscReal time) {
    if (steps >= minimumSimulationStep && time >= minimumSimulationTime) {
        return interval->Check(comm, steps, time);
    }
    return false;
}

#include "registrar.hpp"
REGISTER(ablate::io::interval::Interval, ablate::io::interval::DelayInterval, "Delays the interval until both the simulation time and step criteria is met.",
         ARG(ablate::io::interval::Interval, "interval", "the base interval to check once the requirements are met"),
         OPT(double, "minimumSimulationTime", "minimum simulation time to start checking the specified interval for true (default is zero)"),
         OPT(int, "minimumStep", "minimum step time to start checking the specified interval for true (default is zero)"));
