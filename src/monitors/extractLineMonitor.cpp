#include "extractLineMonitor.hpp"
#include <fstream>
#include <iostream>
#include "environment/runEnvironment.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/petscUtilities.hpp"

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

    // check the size
    int size;
    MPI_Comm_size(flow->GetSubDomain().GetComm(), &size) >> utilities::MpiUtilities::checkError;
    if (size != 1) {
        throw std::runtime_error("The CurveMonitor monitor only works with a single mpi rank");
    }

    // get the cell geom
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;

    // get the min cell size
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(flow->GetSubDomain().GetDM(), NULL, &cellGeomVec, &minCellRadius) >> utilities::PetscUtilities::checkError;
    VecGetDM(cellGeomVec, &dmCell) >> utilities::PetscUtilities::checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;

    PetscMPIInt rank;
    MPI_Comm_rank(flow->GetSubDomain().GetComm(), &rank) >> utilities::MpiUtilities::checkError;

    PetscInt dim;
    DMGetDimension(flow->GetSubDomain().GetDM(), &dim) >> utilities::PetscUtilities::checkError;

    // Now march over each subsegment in the line
    double ds = minCellRadius / 10.0;
    double s = 0.0;
    double L = 0.0;
    std::vector<double> lineVec;
    for (std::size_t d = 0; d < start.size(); d++) {
        L += PetscSqr(end[d] - start[d]);
        lineVec.push_back(end[d] - start[d]);
    }
    L = PetscSqrtReal(L);
    for (auto& c : lineVec) {
        c /= L;
    }

    // Create a location vector
    Vec locVec;
    VecCreateSeq(PETSC_COMM_SELF, dim, &locVec) >> utilities::PetscUtilities::checkError;
    VecSetBlockSize(locVec, dim) >> utilities::PetscUtilities::checkError;

    while (s < L) {
        // Compute the current location
        for (PetscInt d = 0; d < dim; d++) {
            VecSetValue(locVec, d, s * lineVec[d], INSERT_VALUES) >> utilities::PetscUtilities::checkError;
        }
        VecAssemblyBegin(locVec) >> utilities::PetscUtilities::checkError;
        VecAssemblyEnd(locVec) >> utilities::PetscUtilities::checkError;

        // find the point in the mesh
        PetscSF cellSF = NULL;
        DMLocatePoints(flow->GetSubDomain().GetDM(), locVec, DM_POINTLOCATION_NONE, &cellSF) >> utilities::PetscUtilities::checkError;

        const PetscSFNode* cells;
        PetscInt numberFound;
        PetscSFGetGraph(cellSF, NULL, &numberFound, NULL, &cells) >> utilities::PetscUtilities::checkError;
        if (cells[0].rank == rank) {
            // search over the history of indexes
            if (std::find(indexLocations.begin(), indexLocations.end(), cells[0].index) == indexLocations.end()) {
                // we have not counted this cell
                indexLocations.push_back(cells[0].index);

                // get the center location of this cell
                PetscFVCellGeom* cellGeom;
                DMPlexPointLocalRead(dmCell, cells[0].index, cellGeomArray, &cellGeom) >> utilities::PetscUtilities::checkError;
                // figure out where this cell is along the line
                double alongLine = 0.0;
                for (PetscInt d = 0; d < dim; d++) {
                    alongLine += PetscSqr(cellGeom->centroid[d] - start[d]);
                }
                distanceAlongLine.push_back(PetscSqrtReal(alongLine));
            }
        }
        PetscSFDestroy(&cellSF) >> utilities::PetscUtilities::checkError;
        s += ds;
    }
    VecDestroy(&locVec) >> utilities::PetscUtilities::checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> utilities::PetscUtilities::checkError;
}

static PetscErrorCode OutputCurveForField(std::ostream& stream, PetscInt fieldIndex, const ablate::domain::Field& fieldDescription, const std::vector<PetscInt>& indexLocations,
                                          const std::vector<PetscReal> distanceAlongLine, PetscErrorCode(plexPointRead)(DM, PetscInt, PetscInt, const PetscScalar*, void*), Vec u) {
    PetscFunctionBeginUser;
    // Open the array
    const PetscScalar* uArray;
    PetscCall(VecGetArrayRead(u, &uArray));

    // Get the DM for the vec
    DM dm;
    PetscCall(VecGetDM(u, &dm));

    // Output each component
    for (PetscInt c = 0; c < fieldDescription.numberComponents; c++) {
        stream << "#" << fieldDescription.name << (fieldDescription.numberComponents > 1 ? "_" + (fieldDescription.components.empty() ? std::to_string(c) : fieldDescription.components[c]) : "")
               << std::endl;

        // Output each cell
        for (std::size_t i = 0; i < indexLocations.size(); i++) {
            stream << distanceAlongLine[i] << " ";

            // extract the location
            const PetscScalar* values;
            PetscCall(plexPointRead(dm, indexLocations[i], fieldIndex, uArray, &values));

            stream << values[c] << std::endl;
        }
        stream << std::endl;
    }

    PetscCall(VecRestoreArrayRead(u, &uArray));
    PetscFunctionReturn(0);
}

PetscErrorCode ablate::monitors::ExtractLineMonitor::OutputCurve(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx) {
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

            PetscCall(OutputCurveForField(curveFile, fieldIndex, fieldDescription, monitor->indexLocations, monitor->distanceAlongLine, DMPlexPointGlobalFieldRead, u));
        }

        // output each aux variable
        for (const auto& fieldName : monitor->outputAuxFields) {
            auto fieldIndex = flow->GetSubDomain().GetField(fieldName).id;
            const auto& fieldDescription = flow->GetSubDomain().GetField(fieldName);

            PetscCall(
                OutputCurveForField(curveFile, fieldIndex, fieldDescription, monitor->indexLocations, monitor->distanceAlongLine, DMPlexPointLocalFieldRead, flow->GetSubDomain().GetAuxVector()));
        }

        curveFile.close();
    }
    PetscFunctionReturn(0);
}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::ExtractLineMonitor, "Outputs the results along a line as a curve file (beta)", ARG(int, "interval", "output interval"),
         ARG(std::string, "prefix", "the file prefix"), ARG(std::vector<double>, "start", "the line start location"), ARG(std::vector<double>, "end", "the line end location"),
         ARG(std::vector<std::string>, "outputFields", "a list of fields to write to the curve"), ARG(std::vector<std::string>, "outputAuxFields", "a list of aux fields to write to the curve "));
