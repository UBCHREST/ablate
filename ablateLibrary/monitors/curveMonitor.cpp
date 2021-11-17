#include "curveMonitor.hpp"
#include <fstream>
#include "environment/runEnvironment.hpp"
#include "finiteVolume/finiteVolumeSolver.hpp"
#include "io/interval/fixedInterval.hpp"
#include "utilities/mpiError.hpp"

ablate::monitors::CurveMonitor::CurveMonitor(std::shared_ptr<io::interval::Interval> intervalIn, std::string prefix)
    : interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()), filePrefix(prefix) {}

void ablate::monitors::CurveMonitor::Register(std::shared_ptr<solver::Solver> solver) {
    ablate::monitors::Monitor::Register(solver);
    if (!std::dynamic_pointer_cast<ablate::finiteVolume::FiniteVolumeSolver>(solver)) {
        throw std::invalid_argument("The CurveMonitor assumes a FiniteVolumeSolver");
    }
    // This can only be used with a single dimension
    const auto dim = solver->GetSubDomain().GetDimensions();
    if (dim != 1) {
        throw std::invalid_argument("The CurveMonitor monitor can only be used with DMs in 1D");
    }

    // check the size
    int size;
    MPI_Comm_size(solver->GetSubDomain().GetComm(), &size) >> checkMpiError;
    if (size != 1) {
        throw std::runtime_error("The CurveMonitor monitor only works with a single mpi rank");
    }
}

PetscErrorCode ablate::monitors::CurveMonitor::OutputCurve(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx) {
    PetscFunctionBeginUser;

    auto monitor = (ablate::monitors::CurveMonitor*)mctx;
    auto comm = PetscObjectComm((PetscObject)ts);

    if (monitor->interval->Check(comm, steps, time)) {
        // Open a new file
        std::filesystem::path outputFile =
            ablate::environment::RunEnvironment::Get().GetOutputDirectory() / (monitor->filePrefix + monitor->GetSolver()->GetId() + "." + std::to_string(steps) + monitor->fileExtension);
        std::ofstream curveFile;
        curveFile.open(outputFile);

        // March over each solution vector
        curveFile << "#title=" << monitor->GetSolver()->GetId() << std::endl;
        curveFile << "##time=" << time << std::endl << std::endl;

        // build the list of local coordinates
        auto subDM = monitor->GetSolver()->GetSubDomain().GetSubDM();
        Vec cellGeomVec;
        PetscErrorCode ierr = DMPlexGetDataFVM(subDM, nullptr, &cellGeomVec, nullptr, nullptr);

        // March over each cell
        PetscInt cStart, cEnd;
        ierr = DMPlexGetHeightStratum(subDM, 0, &cStart, &cEnd);
        CHKERRQ(ierr);

        try {
            // Output the solution vars
            WriteToCurveFile(curveFile, cStart, cEnd, cellGeomVec, monitor->GetSolver()->GetSubDomain().GetFields(), subDM, monitor->GetSolver()->GetSubDomain().GetSubSolutionVector());

            // check for and then write aux vars
            if (auto auxSubDM = monitor->GetSolver()->GetSubDomain().GetSubAuxDM()) {
                WriteToCurveFile(
                    curveFile, cStart, cEnd, cellGeomVec, monitor->GetSolver()->GetSubDomain().GetFields(domain::FieldLocation::AUX), auxSubDM, monitor->GetSolver()->GetSubDomain().GetSubAuxVector());
            }
        } catch (std::exception& exp) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exp.what());
        }

        curveFile.close();
    }
    PetscFunctionReturn(0);
}
void ablate::monitors::CurveMonitor::WriteToCurveFile(std::ostream& curveFile, PetscInt cStart, PetscInt cEnd, Vec cellGeomVec, const std::vector<domain::Field>& fields, DM dm, Vec valuesVec) {
    DM cellGeomDM;
    VecGetDM(cellGeomVec, &cellGeomDM) >> checkError;
    const PetscScalar* cellGeomArray;
    VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

    // get the value field
    const PetscScalar* valuesArray;
    VecGetArrayRead(valuesVec, &valuesArray) >> checkError;

    for (const auto& field : fields) {
        for (PetscInt comp = 0; comp < field.numberComponents; comp++) {
            curveFile << "#" << field.name << (field.numberComponents > 1 ? "_" + field.components[comp] : "") << std::endl;

            // march over each cell
            for (PetscInt c = cStart; c < cEnd; c++) {
                PetscFVCellGeom* cellGeom;
                DMPlexPointLocalRead(cellGeomDM, c, cellGeomArray, &cellGeom) >> checkError;

                // Now grab the field
                const PetscScalar* values;
                DMPlexPointGlobalFieldRead(dm, c, field.subId, valuesArray, &values) >> checkError;

                if (values) {
                    curveFile << cellGeom->centroid[0] << " ";
                    curveFile << values[comp] << std::endl;
                }
            }
        }
    }
    curveFile << std::endl;
    VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
    VecRestoreArrayRead(valuesVec, &valuesArray) >> checkError;
}

#include "parser/registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::CurveMonitor, "Write 1D results to a curve file", OPT(ablate::io::interval::Interval, "interval", "output interval"),
         OPT(std::string, "prefix", "the file prefix"));
