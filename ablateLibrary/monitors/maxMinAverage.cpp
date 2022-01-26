#include "maxMinAverage.hpp"
#include "io/interval/fixedInterval.hpp"
#include "monitors/logs/stdOut.hpp"

ablate::monitors::MaxMinAverage::MaxMinAverage(const std::string& fieldName, std::shared_ptr<logs::Log> logIn, std::shared_ptr<io::interval::Interval> interval)
    : fieldName(fieldName), log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(interval ? interval : std::make_shared<io::interval::FixedInterval>()) {}

PetscErrorCode ablate::monitors::MaxMinAverage::MonitorMaxMinAverage(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    auto monitor = (ablate::monitors::MaxMinAverage*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        // Get a subvector with only this field
        const auto& field = monitor->GetSolver()->GetSubDomain().GetField(monitor->fieldName);
        IS vecIs;
        Vec vec;
        DM subDm;
        ierr = monitor->GetSolver()->GetSubDomain().GetFieldVector(field, &vecIs, &vec, &subDm);
        CHKERRQ(ierr);

        // get the comm for this monitor
        auto comm = PetscObjectComm((PetscObject)vec);

        // Init the min, max, avg values
        std::vector<double> min(field.numberComponents, std::numeric_limits<double>::max());
        std::vector<double> max(field.numberComponents, std::numeric_limits<double>::lowest());
        std::vector<double> avg(field.numberComponents, 0.0);

        // Get the local size of the vec
        PetscInt locSize;
        ierr = VecGetLocalSize(vec, &locSize);
        CHKERRQ(ierr);

        const PetscScalar* data;
        ierr = VecGetArrayRead(vec, &data);
        CHKERRQ(ierr);

        // Determine the number of data points
        PetscInt pts = locSize / field.numberComponents;

        // Compute max/min/avg values
        for (PetscInt p = 0; p < pts; p++) {
            for (PetscInt d = 0; d < field.numberComponents; d++) {
                const double value = data[p * field.numberComponents + d];
                min[d] = PetscMin(min[d], PetscReal(value));
                max[d] = PetscMax(max[d], PetscReal(value));
                avg[d] += PetscReal(value);
            }
        }

        // Take across all ranks
        std::vector<double> minGlob(field.numberComponents);
        std::vector<double> maxGlob(field.numberComponents);
        std::vector<double> avgGlob(field.numberComponents);

        int mpiError;
        mpiError = MPI_Reduce(&min[0], &minGlob[0], minGlob.size(), MPI_DOUBLE, MPI_MIN, 0, comm);
        CHKERRMPI(mpiError);
        mpiError = MPI_Reduce(&max[0], &maxGlob[0], maxGlob.size(), MPI_DOUBLE, MPI_MAX, 0, comm);
        CHKERRMPI(mpiError);
        mpiError = MPI_Reduce(&avg[0], &avgGlob[0], avgGlob.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
        CHKERRMPI(mpiError);

        // Take the avg
        PetscInt globSize;
        ierr = VecGetSize(vec, &globSize);
        CHKERRQ(ierr);
        for (auto& avgComp : avgGlob) {
            avgComp /= (globSize / field.numberComponents);
        }

        ierr = VecRestoreArrayRead(vec, &data);
        CHKERRQ(ierr);

        // if this is the first time step init the log
        if (!monitor->log->Initialized()) {
            monitor->log->Initialize(comm);
        }

        // Print the results
        monitor->log->Printf("MinMaxAvg %s for timestep %04d:\n", field.name.c_str(), (int)step);
        monitor->log->Print("\tmin", minGlob.size(), &minGlob[0], "%2.3g");
        monitor->log->Print("\n");
        monitor->log->Print("\tmax", maxGlob.size(), &maxGlob[0], "%2.3g");
        monitor->log->Print("\n");
        monitor->log->Print("\tavg", avgGlob.size(), &avgGlob[0], "%2.3g");
        monitor->log->Print("\n");
        ierr = monitor->GetSolver()->GetSubDomain().RestoreFieldVector(field, &vecIs, &vec, &subDm);
        CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::MaxMinAverage, "Prints the min/max/average for a field", ARG(std::string, "field", "the name of the field"),
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"), OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"));
