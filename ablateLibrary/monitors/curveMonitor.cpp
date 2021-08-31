#include "curveMonitor.hpp"
#include <flow/fvFlow.hpp>
#include <fstream>
#include <iostream>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>
#include "environment/runEnvironment.hpp"

ablate::monitors::CurveMonitor::CurveMonitor(int interval, std::string prefix, std::vector<double> start, std::vector<double> end, std::vector<std::string> outputFields,
                                             std::vector<std::string> outputAuxFields)
    : interval(interval), start(start), end(end), outputFields(outputFields), outputAuxFields(outputAuxFields), filePrefix(prefix) {}

void ablate::monitors::CurveMonitor::Register(std::shared_ptr<Monitorable> monitorableObject) {
    // this probe will only work with fV flow and a single process
    flow = std::dynamic_pointer_cast<flow::FVFlow>(monitorableObject);
    if (!flow) {
        throw std::invalid_argument("The CurveMonitor monitor can only be used with ablate::flow::FVFlow");
    }

    // check the size
    int size;
    MPI_Comm_size(PetscObjectComm((PetscObject)flow->GetDM()), &size) >> checkMpiError;
    if (size != 1) {
        throw std::runtime_error("The IgnitionDelay monitor only works with a single mpi rank");
    }

    // get the cell geom
    Vec cellGeomVec;
    DM dmCell;
    const PetscScalar* cellGeomArray;

    // get the min cell size
    PetscReal minCellRadius;
    DMPlexGetGeometryFVM(flow->GetDM(), NULL, &cellGeomVec, &minCellRadius) >> checkError;
    VecGetDM(cellGeomVec, &dmCell) >> checkError;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    PetscMPIInt rank;
    MPI_Comm_rank((PetscObjectComm((PetscObject)flow->GetDM())), &rank) >> checkMpiError;

    PetscInt dim;
    DMGetDimension(flow->GetDM(), &dim) >> checkError;

    // Now march over each sub segment int he line
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
    VecCreateSeq(PETSC_COMM_SELF, dim, &locVec) >> checkError;
    VecSetBlockSize(locVec, dim) >> checkError;

    while (s < L) {
        // Compute the current location
        for (PetscInt d = 0; d < dim; d++) {
            VecSetValue(locVec, d, s * lineVec[d], INSERT_VALUES) >> checkError;
        }
        VecAssemblyBegin(locVec) >> checkError;
        VecAssemblyEnd(locVec) >> checkError;

        // find the point in the mesh
        PetscSF cellSF = NULL;
        DMLocatePoints(flow->GetDM(), locVec, DM_POINTLOCATION_NONE, &cellSF) >> checkError;

        const PetscSFNode* cells;
        PetscInt numberFound;
        PetscSFGetGraph(cellSF, NULL, &numberFound, NULL, &cells) >> checkError;
        if (cells[0].rank == rank) {
            // search over the history of indexes
            if (std::find(indexLocations.begin(), indexLocations.end(), cells[0].index) == indexLocations.end()) {
                // we have not counted this cell
                indexLocations.push_back(cells[0].index);

                // get the center location of this cell
                PetscFVCellGeom* cellGeom;
                DMPlexPointLocalRead(dmCell, cells[0].index, cellGeomArray, &cellGeom) >> checkError;
                // figure out where this cell is along the line
                double alongLine = 0.0;
                for (PetscInt d = 0; d < dim; d++) {
                    alongLine += PetscSqr(cellGeom->centroid[d] - start[d]);
                }
                distanceAlongLine.push_back(PetscSqrtReal(alongLine));
            }
        }
        PetscSFDestroy(&cellSF) >> checkError;
        s += ds;
    }
    VecDestroy(&locVec) >> checkError;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
}

static PetscErrorCode OutputCurveForField(std::ostream& stream, PetscInt fieldIndex, const ablate::flow::FlowFieldDescriptor& fieldDescription, const std::vector<PetscInt>& indexLocations,
                                          const std::vector<PetscReal> distanceAlongLine, PetscErrorCode(plexPointRead)(DM, PetscInt, PetscInt, const PetscScalar*, void*), Vec u) {
    // Open the array
    const PetscScalar* uArray;
    PetscErrorCode ierr = VecGetArrayRead(u, &uArray);
    CHKERRQ(ierr);

    // Get the DM for the vec
    DM dm;
    ierr = VecGetDM(u, &dm);
    CHKERRQ(ierr);

    // Output each component
    for (PetscInt c = 0; c < fieldDescription.components; c++) {
        stream << "#" << fieldDescription.fieldName << (fieldDescription.components > 1 ? "_" + (fieldDescription.componentNames.empty() ? std::to_string(c) : fieldDescription.componentNames[c]) : "")
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

PetscErrorCode ablate::monitors::CurveMonitor::OutputCurve(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr;
    DM dm;
    PetscDS ds;
    ierr = TSGetDM(ts, &dm);
    CHKERRQ(ierr);
    ierr = DMGetDS(dm, &ds);
    CHKERRQ(ierr);

    auto monitor = (ablate::monitors::CurveMonitor*)mctx;
    auto flow = monitor->flow;

    if (steps == 0 || monitor->interval == 0 || (steps % monitor->interval == 0)) {
        // Open a new file
        std::filesystem::path outputFile =
            ablate::environment::RunEnvironment::Get().GetOutputDirectory() / (monitor->filePrefix + "." + std::to_string(monitor->outputIndex) + monitor->fileExtension);
        monitor->outputIndex++;
        std::ofstream curveFile;
        curveFile.open(outputFile);

        // March over each solution vector
        curveFile << "#title=" << flow->GetName() << std::endl;
        curveFile << "##time=" << time << std::endl << std::endl;

        // output each solution variable
        for (const auto& fieldName : monitor->outputFields) {
            auto fieldIndex = flow->GetFieldId(fieldName).value();
            const auto& fieldDescription = flow->GetFieldDescriptor(fieldName);

            ierr = OutputCurveForField(curveFile, fieldIndex, fieldDescription, monitor->indexLocations, monitor->distanceAlongLine, DMPlexPointGlobalFieldRead, u);
            CHKERRQ(ierr);
        }

        // output each aux variable
        for (const auto& fieldName : monitor->outputAuxFields) {
            auto fieldIndex = flow->GetAuxFieldId(fieldName).value();
            const auto& fieldDescription = flow->GetAuxFieldDescriptor(fieldName);

            ierr = OutputCurveForField(curveFile, fieldIndex, fieldDescription, monitor->indexLocations, monitor->distanceAlongLine, DMPlexPointLocalFieldRead, flow->GetAuxField());
            CHKERRQ(ierr);
        }

        curveFile.close();
    }
    CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::CurveMonitor, "Outputs the results along a line as a curve file (beta)", ARG(int, "interval", "output interval"),
         ARG(std::string, "prefix", "the file prefix"), ARG(std::vector<double>, "start", "the line start location"), ARG(std::vector<double>, "end", "the line end location"),
         ARG(std::vector<std::string>, "outputFields", "a list of fields to write to the curve"), ARG(std::vector<std::string>, "outputAuxFields", "a list of aux fields to write to the curve "));
