
#include "probes.hpp"
#include "io/interval/fixedInterval.hpp"
#include "utilities/mpiError.hpp"
#include "utilities/vectorUtilities.hpp"
#include <fstream>

ablate::monitors::Probes::Probes(std::vector<Probe> probes, std::vector<std::string> variableNames, const std::shared_ptr<io::interval::Interval> &intervalIn)
    : allProbes(std::move(probes)), variableNames(std::move(variableNames)), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

ablate::monitors::Probes::Probes(std::vector<std::shared_ptr<Probe>> allProbesPtrs, std::vector<std::string> variableNames, const std::shared_ptr<io::interval::Interval> &interval)
    : Probes(utilities::VectorUtilities::Copy(allProbesPtrs), std::move(variableNames), interval) {}

void ablate::monitors::Probes::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);

    // extract some useful information
    const PetscInt dim = solver->GetSubDomain().GetDimensions();

    {  // Determine what probes live locally
        // extract some useful information
        const auto globalPointsCount = (PetscInt)allProbes.size();

        // Determine which probes live on this rank
        Vec pointVec;
        std::vector<PetscScalar> globalPointsScalar(globalPointsCount * dim);

        // Get a list of all probe locations and convert to a vec
        PetscInt offset = 0;
        for (const auto &probe : allProbes) {
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
        for (std::size_t p = 0; p < allProbes.size(); p++) {
            if (globalProcs[p] == size) {
                throw std::invalid_argument("Cannot locate probe " + allProbes[p].name + " in domain");
            } else if (globalProcs[p] == rank) {
                localProbes.push_back(allProbes[p]);
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

        // Finish the one time set up
        // The redundantPoints flag should not really matter because PETSC_COMM_SELF was used to init the interpolant
        DMInterpolationSetUp(interpolant, solver->GetSubDomain().GetDM(), PETSC_FALSE, PETSC_FALSE) >> checkError;
        interpolants.push_back(interpolant);
    }
}

ablate::monitors::Probes::~Probes() {
    for (auto &interpolant : interpolants) {
        DMInterpolationDestroy(&interpolant) >> checkError;
    }
}

PetscErrorCode ablate::monitors::Probes::UpdateProbes(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx) { return 0; }

ablate::monitors::Probes::ProbeRecorder::ProbeRecorder(int bufferSize, const std::vector<std::string> &variables, std::filesystem::path outputPath) : bufferSize(bufferSize), outputPath(outputPath) {
    // size up the buffer
    buffer = std::vector<std::vector<double>>(bufferSize, std::vector<double>(variables.size()));

    // check to see if the file exists
    if(std::filesystem::exists(outputPath)){
        std::fstream oldFile;
        oldFile.open(outputPath, std::ios::in);
        if (oldFile.is_open()){
            std::string line;
            getline(oldFile, line);
            while(getline(oldFile, line)){
                if
            }
            oldFile.close(); //close the file object.
        }

    }

}
ablate::monitors::Probes::ProbeRecorder::~ProbeRecorder() { WriteBuffer(); }

void ablate::monitors::Probes::ProbeRecorder::AdvanceTime(double time) {}
void ablate::monitors::Probes::ProbeRecorder::SetValue(std::size_t index, double value) {}
void ablate::monitors::Probes::ProbeRecorder::WriteBuffer() {}

#include "registrar.hpp"
REGISTER_DEFAULT(ablate::monitors::Probes::Probe, ablate::monitors::Probes::Probe, "Probe specification struct", ARG(std::string, "name", "name of the probe"),
                 ARG(std::vector<double>, "location", "the probe location"));

REGISTER(ablate::monitors::Monitor, ablate::monitors::Probes, "Records the values of the specified variables at a specific point in space",
         ARG(std::vector<ablate::monitors::Probes::Probe>, "probes", "where to record log (default is stdout)"), ARG(std::vector<std::string>, "variables", "list of variables to output"),
         OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"));
