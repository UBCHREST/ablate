#include "ignitionDelayTemperature.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "monitors/logs/stdOut.hpp"
#include "utilities/mpiUtilities.hpp"
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
    MPI_Comm_size(flow->GetSubDomain().GetComm(), &size) >> utilities::MpiUtilities::checkError;
    if (size != 1) {
        throw std::runtime_error("The IgnitionDelay monitor only works with a single mpi rank");
    }

    // determine the component offset
    computeTemperature = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Temperature, flow->GetSubDomain().GetFields());

    // Convert the location to a vec
    Vec locVec;
    VecCreateSeqWithArray(flow->GetSubDomain().GetComm(), location.size(), location.size(), &location[0], &locVec) >> checkError;

    // Get all points still in this mesh
    PetscSF cellSF = NULL;
    DMLocatePoints(flow->GetSubDomain().GetDM(), locVec, DM_POINTLOCATION_NONE, &cellSF) >> checkError;
    const PetscSFNode* cells;
    PetscInt numberFound;
    PetscMPIInt rank;
    MPI_Comm_rank(flow->GetSubDomain().GetComm(), &rank) >> utilities::MpiUtilities::checkError;

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

    DM dm;
    PetscDS ds;
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMGetDS(dm, &ds));
    PetscInt dim;
    PetscCall(DMGetDimension(dm, &dim));

    // Check for the number of DS, this should be relaxed
    PetscInt numberDS;
    PetscCall(DMGetNumDS(dm, &numberDS));
    if (numberDS > 1) {
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This monitor only supports a single DS in a DM");
    }

    IgnitionDelayTemperature* monitor = (IgnitionDelayTemperature*)ctx;

    // extract the gradLocalVec
    const PetscScalar* uArray;
    PetscCall(VecGetArrayRead(u, &uArray));

    // Get the euler and densityYi values
    const PetscScalar* conserved;
    PetscCall(DMPlexPointGlobalRead(dm, monitor->cellOfInterest, uArray, &conserved));

    // compute the temperature
    double T;
    monitor->computeTemperature.function(conserved, &T, monitor->computeTemperature.context.get());

    // Store the result
    monitor->timeHistory.push_back(crtime);
    monitor->temperatureHistory.push_back(T);

    if (monitor->historyLog) {
        monitor->historyLog->Printf("%" PetscInt_FMT " Time: %g Temperature: %f\n", step, crtime, T);
    }

    PetscCall(VecRestoreArrayRead(u, &uArray));
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::IgnitionDelayTemperature, "Compute the ignition time based upon temperature change",
         ARG(ablate::eos::EOS, "eos", "the eos used to compute temperature"), ARG(std::vector<double>, "location", "the monitor location"),
         ARG(double, "thresholdTemperature", "the temperature used to define ignition delay"), OPT(ablate::monitors::logs::Log, "log", "where to record the final ignition time (default is stdout)"),
         OPT(ablate::monitors::logs::Log, "historyLog", "where to record the time and yi history (default is none)"));
