#include "wallTimeInterval.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::io::interval::WallTimeInterval::WallTimeInterval(int timeInterval, std::function<std::chrono::time_point<std::chrono::system_clock>()> nowFunction)
    : timeInterval(timeInterval), now(nowFunction) {
    previousTime = now();
}
bool ablate::io::interval::WallTimeInterval::Check(MPI_Comm comm, PetscInt steps, PetscReal time) {
    // Get the current duration
    auto nowTime = now();
    auto duration = nowTime - previousTime;

    // If enough wall time las passed
    int checkPoint = duration >= timeInterval;

    // Broadcast to all ranks from root (time can be different on different machines)
    MPI_Bcast(&checkPoint, 1, MPI_INT, 0, comm) >> utilities::MpiUtilities::checkError;

    if (checkPoint) {
        previousTime = nowTime;
    }
    return checkPoint;
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::io::interval::Interval, ablate::io::interval::WallTimeInterval, "Outputs approximately every n wall time seconds", int);
