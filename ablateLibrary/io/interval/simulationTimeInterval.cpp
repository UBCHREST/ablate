#include "simulationTimeInterval.hpp"

ablate::io::interval::SimulationTimeInterval::SimulationTimeInterval(double timeInterval) : timeInterval(timeInterval), nextTime(-1) {}

bool ablate::io::interval::SimulationTimeInterval::Check(MPI_Comm comm, PetscInt steps, PetscReal time) {
    if (time >= nextTime) {
        nextTime = time + timeInterval;
        return true;
    }
    return false;
}

#include "parser/registrar.hpp"
REGISTER_PASS_THROUGH(ablate::io::interval::Interval, ablate::io::interval::SimulationTimeInterval,
                      "Outputs every dt simulation seconds. This will not result in uniform output unless the simulation dt matches.", double);
