#include "ignitionDelayPeakYi.hpp"
#include "finiteVolume/compressibleFlowFields.hpp"
#include "finiteVolume/processes/navierStokesTransport.hpp"
#include "monitors/logs/stdOut.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscError.hpp"

ablate::monitors::IgnitionDelayPeakYi::IgnitionDelayPeakYi(std::string species, std::vector<double> location, std::shared_ptr<logs::Log> logIn, std::shared_ptr<logs::Log> historyLogIn)
    : log(logIn ? logIn : std::make_shared<logs::StdOut>()), historyLog(historyLogIn), species(species), location(location) {}

ablate::monitors::IgnitionDelayPeakYi::~IgnitionDelayPeakYi() {
    // compute the time at the maximum yi
    std::size_t loc = 0;
    double maxValue = 0.0;
    for (std::size_t i = 0; i < yiHistory.size(); i++) {
        if (yiHistory[i] > maxValue) {
            maxValue = yiHistory[i];
            loc = i;
        }
    }
    if (!yiHistory.empty()) {
        log->Printf("Computed Ignition Delay (%s): %g\n", species.c_str(), timeHistory[loc]);
    }
}

void ablate::monitors::IgnitionDelayPeakYi::Register(std::shared_ptr<solver::Solver> monitorableObject) {
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
    eulerId = flow->GetSubDomain().GetField("euler").id;
    yiId = flow->GetSubDomain().GetField("densityYi").id;

    const auto& densityYi = flow->GetSubDomain().GetField("densityYi");
    const auto& speciesList = densityYi.components;
    yiOffset = -1;
    for (std::size_t sp = 0; sp < speciesList.size(); sp++) {
        if (speciesList[sp] == species) {
            yiOffset = sp;
        }
    }
    if (yiOffset < 0) {
        throw std::invalid_argument("The IgnitionDelay monitor cannot find the " + species + " species");
    }

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

PetscErrorCode ablate::monitors::IgnitionDelayPeakYi::MonitorIgnition(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;

    DM dm;
    PetscDS ds;
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMGetDS(dm, &ds));

    // Check for the number of DS, this should be relaxed
    PetscInt numberDS;
    PetscCall(DMGetNumDS(dm, &numberDS));
    if (numberDS > 1) {
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This monitor only supports a single DS in a DM");
    }

    IgnitionDelayPeakYi* monitor = (IgnitionDelayPeakYi*)ctx;

    // extract the gradLocalVec
    const PetscScalar* uArray;
    PetscCall(VecGetArrayRead(u, &uArray));

    // Get the euler and densityYi values
    const PetscScalar* eulerValues;
    const PetscScalar* densityYiValues;
    PetscCall(DMPlexPointGlobalFieldRead(dm, monitor->cellOfInterest, monitor->eulerId, uArray, &eulerValues));
    PetscCall(DMPlexPointGlobalFieldRead(dm, monitor->cellOfInterest, monitor->yiId, uArray, &densityYiValues));

    // Store the result
    double yi = densityYiValues[monitor->yiOffset] / eulerValues[ablate::finiteVolume::CompressibleFlowFields::RHO];
    monitor->timeHistory.push_back(crtime);
    monitor->yiHistory.push_back(yi);

    if (monitor->historyLog) {
        monitor->historyLog->Printf("%" PetscInt_FMT " Time: %g Yi: %f\n", step, crtime, yi);
    }

    PetscCall(VecRestoreArrayRead(u, &uArray));
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::IgnitionDelayPeakYi, "Compute the ignition time based upon peak mass fraction",
         ARG(std::string, "species", "the species used to determine the peak Yi"), ARG(std::vector<double>, "location", "the monitor location"),
         OPT(ablate::monitors::logs::Log, "log", "where to record the final ignition time (default is stdout)"),
         OPT(ablate::monitors::logs::Log, "historyLog", "where to record the time and yi history (default is none)"));
