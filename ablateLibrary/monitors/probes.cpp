#include "probes.hpp"
#include <fstream>
#include <regex>
#include "environment/runEnvironment.hpp"
#include "io/interval/fixedInterval.hpp"
#include "utilities/mpiError.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::monitors::Probes::Probes(const std::shared_ptr<ablate::monitors::probes::ProbeInitializer> &initializer, std::vector<std::string> variableNames,
                                 const std::shared_ptr<io::interval::Interval> &intervalIn, const int bufferSize)
    : initializer(initializer),
      variableNames(std::move(variableNames)),
      interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()),
      bufferSize(bufferSize == 0 ? 100 : bufferSize) {}

void ablate::monitors::Probes::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);

    initializer->Report(solver->GetSubDomain().GetComm());

    // extract some useful information
    const PetscInt dim = solver->GetSubDomain().GetDimensions();

    {  // Determine what probes live locally
        // extract some useful information
        const auto globalPointsCount = (PetscInt)initializer->GetProbes().size();

        // Determine which probes live on this rank
        Vec pointVec;
        std::vector<PetscScalar> globalPointsScalar(globalPointsCount * dim);

        // Get a list of all probe locations and convert to a vec
        PetscInt offset = 0;
        for (const auto &probe : initializer->GetProbes()) {
            // Make sure that the location is at least equal to the number of dims
            if ((PetscInt)probe.location.size() < dim) {
                throw std::invalid_argument("All specified probe locations must be at list dimension " + std::to_string(dim) + ".");
            }
            for (PetscInt d = 0; d < dim; d++) {
                globalPointsScalar[offset++] = probe.location[d];
            }
        }
        VecCreateSeqWithArray(PETSC_COMM_SELF, dim, (PetscInt)globalPointsScalar.size(), globalPointsScalar.data(), &pointVec);

        // Locate the points in the DM
        PetscSF cellSF = nullptr;
        DMLocatePoints(solver->GetSubDomain().GetDM(), pointVec, DM_POINTLOCATION_REMOVE, &cellSF) >> checkMpiError;
        PetscInt numFound;
        const PetscSFNode *foundCells = nullptr;
        const PetscInt *foundPoints = nullptr;
        PetscSFGetGraph(cellSF, nullptr, &numFound, &foundPoints, &foundCells) >> checkMpiError;

        // Let the lowest rank process own each point
        PetscMPIInt rank, size;
        MPI_Comm_rank(solver->GetSubDomain().GetComm(), &rank) >> checkMpiError;
        MPI_Comm_size(solver->GetSubDomain().GetComm(), &size) >> checkMpiError;
        std::vector<PetscMPIInt> foundProcs(globalPointsCount, size);
        std::vector<PetscMPIInt> globalProcs(globalPointsCount, size);

        for (PetscInt p = 0; p < numFound; ++p) {
            if (foundCells[p].index >= 0) {
                foundProcs[foundPoints ? foundPoints[p] : p] = rank;
            }
        }
        // Let the lowest rank process own each point
        MPI_Allreduce(foundProcs.data(), globalProcs.data(), globalPointsCount, MPI_INT, MPI_MIN, solver->GetSubDomain().GetComm()) >> checkMpiError;

        // throw error if location cannot be found and copy over the probes that this rank owns
        for (std::size_t p = 0; p < initializer->GetProbes().size(); p++) {
            if (globalProcs[p] == size) {
                throw std::invalid_argument("Cannot locate probe " + initializer->GetProbes()[p].name + " in domain");
            } else if (globalProcs[p] == rank) {
                localProbes.push_back(initializer->GetProbes()[p]);
            }
        }

        // cleanup
        PetscSFDestroy(&cellSF);
        VecDestroy(&pointVec);
    }

    // Copy over the local points
    std::vector<PetscReal> coordinates(localProbes.size() * dim);
    PetscInt offset = 0;
    for (const auto &probe : localProbes) {
        for (PetscInt d = 0; d < dim; d++) {
            coordinates[offset++] = probe.location[d];
        }
    }

    // convert the variable names to the variable names with components
    PetscInt variableFieldOffset = 0;
    std::vector<std::string> componentNames;

    // Create an interpolant for each variable
    for (const auto &variableName : variableNames) {
        // Get the field information
        const auto &field = solver->GetSubDomain().GetField(variableName);

        // Store this field
        fields.push_back(field);

        // Create the interpolant.  This uses PETSC_COMM_SELF because it should only work over local variables
        DMInterpolationInfo interpolant;
        DMInterpolationCreate(PETSC_COMM_SELF, &interpolant) >> checkError;
        DMInterpolationSetDim(interpolant, dim) >> checkError;
        DMInterpolationSetDof(interpolant, field.numberComponents) >> checkError;

        // Add all local points to the interpolant
        DMInterpolationAddPoints(interpolant, (PetscInt)localProbes.size(), coordinates.data()) >> checkError;

        // Get the subfield dm
        IS subIs;
        DM subDm;
        Vec locVec;
        solver->GetSubDomain().GetFieldLocalVector(field, 0.0, &subIs, &locVec, &subDm) >> checkError;

        // Finish the one time set up
        // The redundantPoints flag should not really matter because PETSC_COMM_SELF was used to init the interpolant
        DMInterpolationSetUp(interpolant, subDm, PETSC_FALSE, PETSC_FALSE) >> checkError;
        interpolants.push_back(interpolant);

        // restore
        solver->GetSubDomain().RestoreFieldLocalVector(field, &subIs, &locVec, &subDm) >> checkError;

        // convert the variable names to the variable names with components
        if (field.numberComponents > 0) {
            for (const auto &componentName : field.components) {
                componentNames.push_back(variableName + "_" + componentName);
            }
        } else {
            componentNames.push_back(variableName);
        }
        fieldOffset.push_back(variableFieldOffset);
        variableFieldOffset += field.numberComponents;
    }

    // Build a ProbeRecorder for each probe
    for (const auto &probe : localProbes) {
        std::filesystem::path probePath = initializer->GetDirectory() / (probe.name + ".csv");
        recorders.emplace_back(bufferSize, componentNames, probePath);
    }
}

ablate::monitors::Probes::~Probes() {
    for (auto &interpolant : interpolants) {
        DMInterpolationDestroy(&interpolant) >> checkError;
    }
}

PetscErrorCode ablate::monitors::Probes::UpdateProbes(TS ts, PetscInt step, PetscReal time, Vec, void *ctx) {
    PetscFunctionBegin;
    auto monitor = (ablate::monitors::Probes *)ctx;
    auto comm = PetscObjectComm((PetscObject)ts);
    PetscErrorCode ierr;

    if (monitor->interval->Check(comm, step, time)) {
        // set the current time for each recorder
        for (auto &recorder : monitor->recorders) {
            recorder.AdvanceTime(time);
        }

        // March over each field
        for (std::size_t it = 0; it < monitor->fields.size(); it++) {
            // determine the field
            const auto &field = monitor->fields[it];

            // Get the sub vector
            IS subIs;
            DM subDm;
            Vec locVec;
            ierr = monitor->GetSolver()->GetSubDomain().GetFieldLocalVector(field, 0.0, &subIs, &locVec, &subDm);
            CHKERRQ(ierr);

            // get a temp vector
            Vec interpValues;
            ierr = DMInterpolationGetVector(monitor->interpolants[it], &interpValues);
            CHKERRQ(ierr);

            // Interpolate
            ierr = DMInterpolationEvaluate(monitor->interpolants[it], subDm, locVec, interpValues);
            CHKERRQ(ierr);

            // Record each value
            const PetscScalar *interValuesArray;
            VecGetArrayRead(interpValues, &interValuesArray);
            PetscInt offset = 0;
            const int &fieldOffset = monitor->fieldOffset[it];
            for (auto &recorder : monitor->recorders) {
                for (PetscInt c = 0; c < field.numberComponents; c++) {
                    recorder.SetValue(fieldOffset + c, interValuesArray[offset++]);
                }
            }

            // restore
            VecRestoreArrayRead(interpValues, &interValuesArray);
            ierr = DMInterpolationRestoreVector(monitor->interpolants[it], &interpValues);
            CHKERRQ(ierr);
            ierr = monitor->GetSolver()->GetSubDomain().RestoreFieldLocalVector(field, &subIs, &locVec, &subDm);
            CHKERRQ(ierr);
        }
    }

    PetscFunctionReturn(0);
}

ablate::monitors::Probes::ProbeRecorder::ProbeRecorder(int bufferSizeIn, const std::vector<std::string> &variables, const std::filesystem::path &outputPath)
    : bufferSize(PetscMax(bufferSizeIn, 1)), outputPath(outputPath) {
    // size up the buffers
    buffer = std::vector<std::vector<double>>(bufferSize, std::vector<double>(variables.size()));
    timeHistory = std::vector<double>(bufferSize);

    // check to see if the file exists
    if (std::filesystem::exists(outputPath)) {
        // build regex to get the first number column
        const auto regex = std::regex("^([0-9.eE-]*)");

        std::fstream oldFile;
        oldFile.open(outputPath, std::ios::in);
        if (oldFile.is_open()) {
            std::string line;
            getline(oldFile, line);
            while (getline(oldFile, line)) {
                std::smatch m;
                regex_search(line, m, regex);
                for (auto x : m) {
                    lastOutputTime = PetscMax(lastOutputTime, std::stod(x));
                }
            }
            oldFile.close();  // close the file object.
        }
    } else {
        // write the header file
        std::ofstream probeFile;
        probeFile.open(outputPath);
        probeFile << "time,";
        for (const auto &variable : variables) {
            probeFile << variable << ",";
        }
        probeFile << std::endl;
        probeFile.close();
    }
}
ablate::monitors::Probes::ProbeRecorder::~ProbeRecorder() { WriteBuffer(); }

void ablate::monitors::Probes::ProbeRecorder::AdvanceTime(double time) {
    if (time > lastOutputTime) {
        if (activeIndex + 1 >= bufferSize) {
            WriteBuffer();
        }

        activeIndex++;
        lastOutputTime = time;
        timeHistory[activeIndex] = time;
    }
}

void ablate::monitors::Probes::ProbeRecorder::SetValue(std::size_t index, double value) {
    if (activeIndex >= 0) {
        buffer[activeIndex][index] = value;
    }
}

void ablate::monitors::Probes::ProbeRecorder::WriteBuffer() {
    // write the header file
    std::ofstream probeFile;
    probeFile.open(outputPath, std::ios_base::app);
    for (int r = 0; r <= activeIndex; r++) {
        probeFile << timeHistory[r] << ",";
        for (const auto &value : buffer[r]) {
            probeFile << value << ",";
        }
        probeFile << "\n";
    }
    probeFile.close();
    activeIndex = -1;
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::Probes, "Records the values of the specified variables at a specific point in space",
         ARG(ablate::monitors::probes::ProbeInitializer, "probes", "where to record log (default is stdout)"), ARG(std::vector<std::string>, "variables", "list of variables to output"),
         OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"), OPT(int, "bufferSize", "how often the probe file is written (default is 100, must be > 0)"));
