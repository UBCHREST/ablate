#include "extractLineMonitor.hpp"
#include <fstream>
#include <iostream>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>
#include "environment/runEnvironment.hpp"

ablate::monitors::ExtractLineMonitor::ExtractLineMonitor(int interval, std::string prefix, std::vector<double> start, std::vector<double> end, std::vector<std::string> outputFields,
                                                         std::vector<std::string> outputAuxFields)
    : interval(interval), start(start), end(end), outputFields(outputFields), outputAuxFields(outputAuxFields), filePrefix(prefix) {}

void ablate::monitors::ExtractLineMonitor::Register(std::shared_ptr<solver::Solver> monitorableObject) {
    ablate::monitors::Monitor::Register(monitorableObject);

    // this probe will only work with fV flow and a single process
    flow = std::dynamic_pointer_cast<finiteVolume::FiniteVolumeSolver>(monitorableObject);
    if (!flow) {
        throw std::invalid_argument("The ExtractLineMonitor monitor can only be used with ablate::finiteVolume::FiniteVolume");
    }

    // check the size of the subdomain for which the line monitor has been called?
    int size;
    MPI_Comm_size(flow->GetSubDomain().GetComm(), &size) >> checkMpiError;
    if (size != 1) {
        throw std::runtime_error("The CurveMonitor monitor only works with a single mpi rank");
    }

    // get the cell geom
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;

    // get the min cell size
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(flow->GetSubDomain().GetDM(), NULL, &cellGeomVec, &minCellRadius) >> checkError;
    VecGetDM(cellGeomVec, &dmCell) >> checkError; //Gets the DM defining the data layout of the vector
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    PetscMPIInt rank;
    MPI_Comm_rank(flow->GetSubDomain().GetComm(), &rank) >> checkMpiError;

    PetscInt dim;
    DMGetDimension(flow->GetSubDomain().GetDM(), &dim) >> checkError; //Gets the number of dimensions which the domain exists in?

    // Now march over each subsegment in the line
    double ds = minCellRadius / 10.0; //Sets the minimum step size which the line monitor is allowed to take
    double s = 0.0; //s represents how far the line has been marched through?
    double L = 0.0; //L represents the length of the line?
    std::vector<double> lineVec; //Vector in the direction of the line
    for (std::size_t d = 0; d < start.size(); d++) { //
        L += PetscSqr(end[d] - start[d]); //PetscSqr: squares a number?
        lineVec.push_back(end[d] - start[d]); //What is push_back? Adds a new element to the vector
    }
    L = PetscSqrtReal(L); //Takes the square root of the sum of differences?
    for (auto& c : lineVec) {
        c /= L; //TODO: What does c represent?
    }

    // Create a location vector (Location of what? The current cell?)
    Vec locVec; //Vector representing the location of a cell?
    VecCreateSeq(PETSC_COMM_SELF, dim, &locVec) >> checkError; //Creates the vector with a length equal to the number of dimensions
    VecSetBlockSize(locVec, dim) >> checkError; //TODO: What is vec set block size? Probably something related to memory management

    while (s < L) { //While the vector traveled distance is less than the line length total?
        // Compute the current location
        for (PetscInt d = 0; d < dim; d++) { //For each index in the vector? (For 3d, this would be three-dimensional, therefore this writes the location into a petsc vector)
            VecSetValue(locVec, d, s * lineVec[d], INSERT_VALUES) >> checkError; //Insert s*lineVec[d] into the locVec vector at index d
        }
        VecAssemblyBegin(locVec) >> checkError; //Initiates the MPI passing for nonlocal operations (execute the index writing)
        VecAssemblyEnd(locVec) >> checkError; //Same as above

        // find the point in the mesh
        PetscSF cellSF = NULL; //PETSc object for setting up and managing the communication of certain entries of arrays and Vecs between MPI processes.
        DMLocatePoints(flow->GetSubDomain().GetDM(), locVec, DM_POINTLOCATION_NONE, &cellSF) >> checkError; //Locate the points in v in the mesh and return a PetscSF of the containing cells

        const PetscSFNode* cells;
        PetscInt numberFound; //Number of what found? Points
        PetscSFGetGraph(cellSF, NULL, &numberFound, NULL, &cells) >> checkError; //Get the graph specifying a parallel star forest //TODO: what is a parallel star forest exactly?
        if (cells[0].rank == rank) {
            // search over the history of indexes
            if (std::find(indexLocations.begin(), indexLocations.end(), cells[0].index) == indexLocations.end()) {
                // we have not counted this cell
                indexLocations.push_back(cells[0].index);

                // get the center location of this cell //TODO: This can be used in order to find the cell locations based on index?
                PetscFVCellGeom* cellGeom;
                DMPlexPointLocalRead(dmCell, cells[0].index, cellGeomArray, &cellGeom) >> checkError;
                // figure out where this cell is along the line
                double alongLine = 0.0;
                for (PetscInt d = 0; d < dim; d++) {
                    alongLine += PetscSqr(cellGeom->centroid[d] - start[d]);
                }
                distanceAlongLine.push_back(PetscSqrtReal(alongLine)); //Adds the distance that this cell step has traveled?
            }
        }
        PetscSFDestroy(&cellSF) >> checkError;
        s += ds;
    }
    VecDestroy(&locVec) >> checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
}

static PetscErrorCode OutputCurveForField(std::ostream& stream, PetscInt fieldIndex, const ablate::domain::Field& fieldDescription, const std::vector<PetscInt>& indexLocations,
                                          const std::vector<PetscReal> distanceAlongLine, PetscErrorCode(plexPointRead)(DM, PetscInt, PetscInt, const PetscScalar*, void*), Vec u) {
    PetscFunctionBeginUser;
    // Open the array
    const PetscScalar* uArray;
    PetscErrorCode ierr = VecGetArrayRead(u, &uArray);
    CHKERRQ(ierr);

    // Get the DM for the vec
    DM dm;
    ierr = VecGetDM(u, &dm);
    CHKERRQ(ierr);

    // Output each component
    for (PetscInt c = 0; c < fieldDescription.numberComponents; c++) {
        stream << "#" << fieldDescription.name << (fieldDescription.numberComponents > 1 ? "_" + (fieldDescription.components.empty() ? std::to_string(c) : fieldDescription.components[c]) : "")
               << std::endl;

        // Output each cell
        for (std::size_t i = 0; i < indexLocations.size(); i++) {
            stream << distanceAlongLine[i] << " ";

            // extract the location
            const PetscScalar* values;
            ierr = plexPointRead(dm, indexLocations[i], fieldIndex, uArray, &values);
            CHKERRQ(ierr);

            stream << values[c] << std::endl;
        }
        stream << std::endl;
    }

    ierr = VecRestoreArrayRead(u, &uArray);
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::monitors::ExtractLineMonitor::OutputCurve(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    DM dm;
    PetscDS ds;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    // Check for the number of DS, this should be relaxed
    PetscInt numberDS;
    ierr = DMGetNumDS(dm, &numberDS);
    CHKERRQ(ierr);
    if (numberDS > 1) {
        SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "This monitor only supports a single DS in a DM");
    }

    auto monitor = (ablate::monitors::ExtractLineMonitor*)mctx;
    auto flow = monitor->flow;

    if (steps == 0 || monitor->interval == 0 || (steps % monitor->interval == 0)) {
        // Open a new file
        std::filesystem::path outputFile =
            ablate::environment::RunEnvironment::Get().GetOutputDirectory() / (monitor->filePrefix + "." + std::to_string(monitor->outputIndex) + monitor->fileExtension);
        monitor->outputIndex++;
        std::ofstream curveFile;
        curveFile.open(outputFile);

        // March over each solution vector
        curveFile << "#title=" << flow->GetSolverId() << std::endl;
        curveFile << "##time=" << time << std::endl << std::endl;

        // output each solution variable
        for (const auto& fieldName : monitor->outputFields) {
            auto fieldIndex = flow->GetSubDomain().GetField(fieldName).id;
            const auto& fieldDescription = flow->GetSubDomain().GetField(fieldName);

            ierr = OutputCurveForField(curveFile, fieldIndex, fieldDescription, monitor->indexLocations, monitor->distanceAlongLine, DMPlexPointGlobalFieldRead, u);
            CHKERRQ(ierr);
        }

        // output each aux variable
        for (const auto& fieldName : monitor->outputAuxFields) {
            auto fieldIndex = flow->GetSubDomain().GetField(fieldName).id;
            const auto& fieldDescription = flow->GetSubDomain().GetField(fieldName);

            ierr = OutputCurveForField(curveFile, fieldIndex, fieldDescription, monitor->indexLocations, monitor->distanceAlongLine, DMPlexPointLocalFieldRead, flow->GetSubDomain().GetAuxVector());
            CHKERRQ(ierr);
        }

        curveFile.close();
    }
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::ExtractLineMonitor, "Outputs the results along a line as a curve file (beta)", ARG(int, "interval", "output interval"),
         ARG(std::string, "prefix", "the file prefix"), ARG(std::vector<double>, "start", "the line start location"), ARG(std::vector<double>, "end", "the line end location"),
         ARG(std::vector<std::string>, "outputFields", "a list of fields to write to the curve"), ARG(std::vector<std::string>, "outputAuxFields", "a list of aux fields to write to the curve "));
