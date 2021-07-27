#include "ignitionDelayTemperature.hpp"
#include "flow/fvFlow.hpp"
#include "flow/processes/eulerAdvection.hpp"
#include "monitors/logs/stdOut.hpp"
#include "utilities/mpiError.hpp"
#include "utilities/petscError.hpp"

ablate::monitors::IgnitionDelayTemperature::IgnitionDelayTemperature(std::shared_ptr<eos::EOS> eosIn, std::vector<double> location, double thresholdTemperatureIn, std::shared_ptr<logs::Log> logIn,
                                                                     std::shared_ptr<logs::Log> historyLogIn)
    : eos(eosIn), thresholdTemperature(thresholdTemperatureIn), log(logIn ? logIn : std::make_shared<logs::StdOut>()), historyLog(historyLogIn), location(location) {}

ablate::monitors::IgnitionDelayTemperature::~IgnitionDelayTemperature() {
    for (std::size_t i = 0; i < temperatureHistory.size(); i++) {
        if (temperatureHistory[i] > thresholdTemperature) {
            log->Printf("Computed Ignition Delay (Temperature): %f\n", timeHistory[i]);
            return;
        }
    }
}

void ablate::monitors::IgnitionDelayTemperature::Register(std::shared_ptr<Monitorable> monitorableObject) {
    // this probe will only work with fV flow with a single mpi rank for now.  It should be replaced with DMInterpolationEvaluate
    auto flow = std::dynamic_pointer_cast<ablate::flow::FVFlow>(monitorableObject);
    if (!flow) {
        throw std::invalid_argument("The IgnitionDelay monitor can only be used with ablate::flow::FVFlow");
    }

    // check the size
    int size;
    MPI_Comm_size(PetscObjectComm((PetscObject)flow->GetDM()), &size) >> checkMpiError;
    if (size != 1) {
        throw std::runtime_error("The IgnitionDelay monitor only works with a single mpi rank");
    }

    // determine the component offset
    auto eulerIdValue = flow->GetFieldId("euler");
    if (eulerIdValue) {
        eulerId = eulerIdValue.value();
    } else {
        throw std::invalid_argument("The IgnitionDelay monitor expects to find euler in the ablate::flow::FVFlow");
    }

    auto densityYiIdValue = flow->GetFieldId("densityYi");
    if (densityYiIdValue) {
        yiId = densityYiIdValue.value();
    } else {
        throw std::invalid_argument("The IgnitionDelay monitor expects to find densityYi in the ablate::flow::FVFlow");
    }

    // Locate the closest cell
    PetscReal distance = PETSC_MAX_REAL;

    // Get the cell start and end for the fv cells
    PetscInt cellStart, cellEnd;
    DMPlexGetHeightStratum(flow->GetDM(), 0, &cellStart, &cellEnd) >> checkError;

    // Extract the cell geometry, and the dm that holds the information
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;
    DMPlexGetGeometryFVM(flow->GetDM(), NULL, &cellGeomVec, NULL) >> checkError;
    VecGetDM(cellGeomVec, &dmCell) >> checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    for (PetscInt c = cellStart; c < cellEnd; ++c) {
        PetscFVCellGeom* cellGeom;
        DMPlexPointLocalRead(dmCell, c, cellGeomArray, &cellGeom) >> checkError;

        PetscReal dis = 0.0;
        for (std::size_t d = 0; d < location.size(); d++) {
            dis += PetscSqr(cellGeom->centroid[d] - location[d]);
        }
        dis = PetscSqrtReal(dis);
        if (dis < distance) {
            cellOfInterest = c;
            distance = dis;
        }
    }
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    // init the log(s)
    log->Initialize(PetscObjectComm((PetscObject)flow->GetDM()));
    if (historyLog) {
        historyLog->Initialize(PetscObjectComm((PetscObject)flow->GetDM()));
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
    const double density = eulerValues[ablate::flow::processes::EulerAdvection::RHO];
    monitor->eos->GetComputeTemperatureFunction()(dim,
                                                  density,
                                                  eulerValues[ablate::flow::processes::EulerAdvection::RHOE] / density,
                                                  eulerValues + ablate::flow::processes::EulerAdvection::RHOU,
                                                  densityYiValues,
                                                  &T,
                                                  monitor->eos->GetComputeTemperatureContext());

    // Store the result
    monitor->timeHistory.push_back(crtime);
    monitor->temperatureHistory.push_back(T);

    if (monitor->historyLog) {
        monitor->historyLog->Printf("%d Time: %f Temperature: %f\n", step, crtime, T);
    }

    ierr = VecRestoreArrayRead(u, &uArray);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::IgnitionDelayTemperature, "Compute the ignition time based upon temperature change", ARG(eos::EOS, "eos", "the eos used to compute temperature"),
         ARG(std::vector<double>, "location", "the monitor location"), ARG(double, "thresholdTemperature", "the temperature used to define ignition delay"),
         OPT(ablate::monitors::logs::Log, "log", "where to record the final ignition time (default is stdout)"),
         OPT(ablate::monitors::logs::Log, "historyLog", "where to record the time and yi history (default is none)"));
