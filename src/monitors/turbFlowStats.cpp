#include "monitors/turbFlowStats.hpp"
#include "monitors/logs/stdOut.hpp"
#include "io/interval/fixedInterval.hpp"
#include "petscmath.h"
#define tiny 1e-30

ablate::monitors::TurbFlowStats::TurbFlowStats(const std::string nameIn, std::shared_ptr<logs::Log> logIn, std::shared_ptr<io::interval::Interval> intervalIn)
    : fieldName(nameIn), log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()){}

PetscErrorCode ablate::monitors::TurbFlowStats::MonitorTurbFlowStats(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;

    //Loads in context
    auto monitor = (ablate::monitors::TurbFlowStats*) ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        const auto& field = monitor -> GetSolver()->GetSubDomain().GetField(monitor -> fieldName);
        IS vecIS;
        DM subDM;
        Vec vec;
        ierr = monitor->GetSolver()->GetSubDomain().GetFieldGlobalVector(field, &vecIS, &vec, &subDM);CHKERRQ(ierr);
        MPI_Comm comm = PetscObjectComm((PetscObject)vec);
        if (monitor->log->Initialized()) {
            monitor->log->Initialize(comm);
        }

        //Allocate memory for the sum and square sum. Need a vector due to multiple field components per field (eg velocity)
        std::vector<double> sum(field.numberComponents, 0.0);
        std::vector<double> sum2(field.numberComponents, 0.0);

        //Get the local size
        PetscInt localSize;
        ierr = VecGetLocalSize(vec, &localSize); CHKERRQ(ierr);

        //Get a pointer to the array data. Data is stored in repeating pattern according to components (see application below)
        const PetscScalar * data;
        ierr = VecGetArrayRead(vec, &data); CHKERRQ(ierr);

        //Find the number of local points in the data space
        PetscInt pts = localSize / field.numberComponents;

        //Find the sum and square sum locally
        for(PetscInt p = 0; p < pts; p++) {
            for(PetscInt d; d < field.numberComponents; d++) {
                const double value = data[p*field.numberComponents + d];
                sum[d] += value;
                sum2[d] += value * value;
            }
        }

        //Sum data across all processors
        std::vector<double> sumGlob(field.numberComponents);
        std::vector<double> sum2Glob(field.numberComponents);

        int mpiError;
        mpiError = MPI_Reduce(&sumGlob[0], &sum[0], sumGlob.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
        CHKERRMPI(mpiError);
        mpiError = MPI_Reduce(&sum2Glob[0], &sum2[0], sum2Glob.size(), MPI_DOUBLE, MPI_SUM, 0, comm);
        CHKERRMPI(mpiError);

        //Calculate RMS of data
        PetscInt globSize;
        ierr = VecGetSize(vec, &globSize); CHKERRQ(ierr);
        std::vector<double> rms(field.numberComponents);

        for(PetscInt d = 0; d < field.numberComponents; d++) {
            rms[d] = sum2[d] / (globSize + tiny) - PetscPowReal(sum[d] / (globSize + tiny), 2);
            rms[d] = PetscSqrtReal(rms[d]);
        }

        //Output the values to the log
        monitor->log->Printf("RMS of %s at timestep %d: ", monitor->fieldName.c_str(), (int)step);
        monitor->log->Print("\t", rms.size(), &rms[0], "%3.2g");
        monitor->log->Print("\n");
    }


    PetscFunctionReturn(0);
}

#include <registrar.hpp>
REGISTER(ablate::monitors::Monitor, ablate::monitors::TurbFlowStats, "Computes turbulent flow statistics", ARG(std::string, "field", "The name of the field"),
         OPT(ablate::monitors::logs::Log, "log", "Where the data will be sent (default is stdout)"), OPT(ablate::io::interval::Interval, "interval", "The monitor output interval"));