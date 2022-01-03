#include "ignitionDelayTemperature.hpp"
#include "finiteVolume/processes/eulerTransport.hpp"
#include "monitors/logs/stdOut.hpp"
#include "utilities/mpiError.hpp"
#include "utilities/petscError.hpp"

ablate::monitors::IgnitionDelayTemperature::IgnitionDelayTemperature(std::shared_ptr<eos::EOS> eosIn, std::vector<double> location, double thresholdTemperatureIn, std::shared_ptr<logs::Log> logIn,
                                                                     std::shared_ptr<logs::Log> historyLogIn)
    : eos(eosIn), thresholdTemperature(thresholdTemperatureIn), log(logIn ? logIn : std::make_shared<logs::StdOut>()), historyLog(historyLogIn), location(location) {}

ablate::monitors::IgnitionDelayTemperature::~IgnitionDelayTemperature() {
    for (std::size_t i = 0; i < temperatureHistory.size(); i++) {
        if (temperatureHistory[i] > thresholdTemperature) {
            log->Printf("Computed Ignition Delay (Temperature): %g\n", timeHistory[i]);
            return;
        }
    }
}

void ablate::monitors::IgnitionDelayTemperature::Register(std::shared_ptr<solver::Solver> monitorableObject) {
    ablate::monitors::Monitor::Register(monitorableObject);

    // this probe will only work with fV flow with a single mpi rank for now.  It should be replaced with DMInterpolationEvaluate
    auto flow = std::dynamic_pointer_cast<ablate::finiteVolume::FiniteVolumeSolver>(monitorableObject);
    if (!flow) {
        throw std::invalid_argument("The IgnitionDelay monitor can only be used with ablate::finiteVolume::FiniteVolume");
    }

    // check the size
    int size;
    MPI_Comm_size(flow->GetSubDomain().GetComm(), &size) >> checkMpiError;
    if (size != 1) {
        throw std::runtime_error("The IgnitionDelay monitor only works with a single mpi rank");
    }

    // determine the component offset
    eulerId = flow->GetSubDomain().GetField("euler").id;
    yiId = flow->GetSubDomain().GetField("densityYi").id;

    // Convert the location to a vec
    Vec locVec;
    VecCreateSeqWithArray(flow->GetSubDomain().GetComm(), location.size(), location.size(), &location[0], &locVec) >> checkError;

    // Get all points still in this mesh
    PetscSF cellSF = NULL;
    DMLocatePoints(flow->GetSubDomain().GetDM(), locVec, DM_POINTLOCATION_NONE, &cellSF) >> checkError;
    const PetscSFNode* cells;
    PetscInt numberFound;
    PetscMPIInt rank;
    MPI_Comm_rank(flow->GetSubDomain().GetComm(), &rank) >> checkMpiError;

    PetscSFGetGraph(cellSF, NULL, &numberFound, NULL, &cells) >> checkError;
    if (numberFound == 1) {
        if (cells[0].rank == rank) {
            cellOfInterest = cells[0].index;
        }
    } else {
        throw std::runtime_error("Cannot locate cell for location in IgnitionDelayPeakYi");
    }

    // restore
    PetscSFDestroy(&cellSF) >> checkError;
    VecDestroy(&locVec) >> checkError;

    // init the log(s)
    log->Initialize(flow->GetSubDomain().GetComm());
    if (historyLog) {
        historyLog->Initialize(flow->GetSubDomain().GetComm());
    }
}
PetscErrorCode ablate::monitors::IgnitionDelayTemperature::MonitorIgnition(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    DM dm;
    PetscDS ds;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);
    PetscInt dim;
    ierr = DMGetDimension(dm, &dim);
    CHKERRQ(ierr);

    // Check for the number of DS, this should be relaxed
    PetscInt numberDS;
    ierr = DMGetNumDS(dm, &numberDS);
    CHKERRQ(ierr);
    if (numberDS > 1) {
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This monitor only supports a single DS in a DM");
    }

    IgnitionDelayTemperature* monitor = (IgnitionDelayTemperature*)ctx;

    // extract the gradLocalVec
    const PetscScalar* uArray;
    ierr = VecGetArrayRead(u, &uArray);
    CHKERRQ(ierr);

    // Get the euler and densityYi values
    const PetscScalar* eulerValues;
    const PetscScalar* densityYiValues;
    ierr = DMPlexPointGlobalFieldRead(dm, monitor->cellOfInterest, monitor->eulerId, uArray, &eulerValues);
    CHKERRQ(ierr);
    ierr = DMPlexPointGlobalFieldRead(dm, monitor->cellOfInterest, monitor->yiId, uArray, &densityYiValues);
    CHKERRQ(ierr);

    // compute the temperature
    // using ComputeTemperatureFunction = PetscErrorCode (*)(PetscInt dim, PetscReal density, PetscReal totalEnergy, const PetscReal* massFlux, const PetscReal densityYi[], PetscReal* T, void* ctx);
    double T;
    const double density = eulerValues[ablate::finiteVolume::processes::FlowProcess::RHO];
    monitor->eos->GetComputeTemperatureFunction()(dim,
                                                  density,
                                                  eulerValues[ablate::finiteVolume::processes::FlowProcess::RHOE] / density,
                                                  eulerValues + ablate::finiteVolume::processes::FlowProcess::RHOU,
                                                  densityYiValues,
                                                  &T,
                                                  monitor->eos->GetComputeTemperatureContext());

    // Store the result
    monitor->timeHistory.push_back(crtime);
    monitor->temperatureHistory.push_back(T);

    if (monitor->historyLog) {
        monitor->historyLog->Printf("%d Time: %g Temperature: %f\n", step, crtime, T);
    }

    ierr = VecRestoreArrayRead(u, &uArray);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::IgnitionDelayTemperature, "Compute the ignition time based upon temperature change",
         ARG(ablate::eos::EOS, "eos", "the eos used to compute temperature"), ARG(std::vector<double>, "location", "the monitor location"),
         ARG(double, "thresholdTemperature", "the temperature used to define ignition delay"), OPT(ablate::monitors::logs::Log, "log", "where to record the final ignition time (default is stdout)"),
         OPT(ablate::monitors::logs::Log, "historyLog", "where to record the time and yi history (default is none)"));
