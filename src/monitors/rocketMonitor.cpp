#include "rocketMonitor.hpp"
#include <utility>
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>
#include "io/interval/fixedInterval.hpp"
#include "monitors/logs/stdOut.hpp"
#include "monitors/logs/log.hpp"
#include "monitor.hpp"

ablate::monitors::RocketMonitor::RocketMonitor(const std::string nameIn, std::shared_ptr<domain::Region> regionIn, std::shared_ptr<domain::Region> fieldBoundaryIn, std::shared_ptr<eos::EOS> eosIn, const std::shared_ptr<logs::Log>& logIn, const std::shared_ptr<io::interval::Interval>& intervalIn)
    : name(nameIn), region(std::move(regionIn)), fieldBoundary(std::move(fieldBoundaryIn)), eos(std::move(eosIn)), log(logIn ? logIn : std::make_shared<logs::StdOut>()), interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()) {}

void ablate::monitors::RocketMonitor::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);
//    PetscMPIInt rank;
//    MPI_Comm_rank(solver->GetSubDomain().GetComm(), &rank) >> checkMpiError;
//    PetscInt dim;
//    DMGetDimension(solver->GetSubDomain().GetDM(), &dim) >> checkError;
}

PetscErrorCode ablate::monitors::RocketMonitor::OutputRocket(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::RocketMonitor*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {

        auto dm = monitor->GetSolver()->GetSubDomain().GetDM(); // get the dm
        auto solDM = monitor->GetSolver()->GetSubDomain().GetDM(); // get the sol dm
        auto comm = PetscObjectComm((PetscObject)monitor->GetSolver()->GetSubDomain().GetDM()); // The communicator in which the reduction takes place.

        PetscInt dim;
        DMGetDimension(dm, &dim); // get the dimensions of the dm
        PetscMPIInt bufferSize; // initialize buffer size
        PetscMPIIntCast(dim, &bufferSize); // define the number of elements in buffer
        // check to see if there is a ghost label
        DMLabel ghostLabel;
        DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

        const auto& fieldEuler = monitor->GetSolver()->GetSubDomain().GetField("euler"); // get the euler field
        PetscReal* cellEuler;
        PetscReal* conservedValues;
        PetscReal cellPressure;

        const auto auxVec = monitor->GetSolver()->GetSubDomain().GetAuxVector();
        const PetscScalar* auxArray;
        VecGetArrayRead(auxVec, &auxArray);
        const auto solVec = monitor->GetSolver()->GetSubDomain().GetSolutionVector();
        const PetscScalar* solArray;
        VecGetArrayRead(solVec, &solArray);

        Vec faceGeomVec;
        Vec cellGeomVec;
        DMPlexComputeGeometryFVM(dm, &cellGeomVec, &faceGeomVec) >> checkError;
        DM faceDM;
        DM cellDM;

        VecGetDM(faceGeomVec, &faceDM) >> checkError;
        const PetscScalar* faceGeomArray;
        VecGetArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
        VecGetDM(cellGeomVec, &cellDM) >> checkError;
        const PetscScalar* cellGeomArray;
        VecGetArrayRead(cellGeomVec, &cellGeomArray) >> checkError;

        PetscReal tol = 1e-3;

        // initialize vectors (send_buffer)
        PetscReal mDotCell[3] = {0,0,0};
        PetscReal mDotTotal[3] = {0,0,0};
        PetscReal thrustCell[3] = {0,0,0};
        PetscReal thrustTotal[3] = {0,0,0};

        // initialize global vectors (receive_buffer)
        PetscReal mDotTotalGlob[3] = {0,0,0};
        PetscReal thrustTotalGlob[3] = {0,0,0};
        PetscReal IspGlob[3] = {0,0,0};

        // find all faces
        PetscInt fStart, fEnd;
        DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> checkError;

        /** Get the current rank associated with this process */
        PetscMPIInt rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.


        // need to look for faces at specified fieldBoundary then find cells bordering those faces which are in specified region
        for (PetscInt face = fStart; face < fEnd; ++face) { // Iterate through all faces to check if in fieldBoundary
            if (ablate::domain::Region::InRegion(monitor->fieldBoundary, dm, face)) { // Check if each face is in fieldBoundary
                PetscFVFaceGeom* fg;
                DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError; // read face geometry for face

                PetscInt numberNeighborCells;
                const PetscInt* neighborCells;
                DMPlexGetSupportSize(dm, face, &numberNeighborCells) >> ablate::checkError;
                DMPlexGetSupport(dm, face, &neighborCells) >> ablate::checkError;
                for (PetscInt n = 0; n < numberNeighborCells; n++) {

                    // Make sure that we are not working with a ghost cell
                    PetscInt ghost = -1;
                    if (ghostLabel) {
                        DMLabelGetValue(ghostLabel, neighborCells[n], &ghost);
                    }
                    if (ghost >= 0) {
                        continue;
                    }

                    if (ablate::domain::Region::InRegion(monitor->region, dm, neighborCells[n])) {  // check if cell is in region

                        DMPlexPointLocalRead(solDM, neighborCells[n], solArray, &conservedValues);  // Retrieve conserved values from cell
                        monitor->computePressure = monitor->eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Pressure, monitor->GetSolver()->GetSubDomain().GetFields()); // get decode state function/context
                        monitor->computePressure.function(conservedValues, &cellPressure, monitor->computePressure.context.get()); // Retrieve pressure from cell
                        DMPlexPointLocalFieldRead(solDM, neighborCells[n], fieldEuler.id, solArray, &cellEuler);  // retrieve euler field for density, density*velocity

                        for (PetscInt d = 0; d < dim; d++) {
                            mDotCell[d] = fg->normal[d]*cellEuler[finiteVolume::CompressibleFlowFields::RHOU+d]; // calculate mass flow rate for the cell
                            mDotTotal[d] = mDotTotal[d] + mDotCell[d]; // summation of total mass flow rate along fieldBoundary
                            if (abs(mDotCell[d]) > tol) {
                                thrustCell[d] = (mDotCell[d])*((cellEuler[finiteVolume::CompressibleFlowFields::RHOU+d])/(cellEuler[finiteVolume::CompressibleFlowFields::RHO])) + (fg->normal[d])*(cellPressure-101325); // calculate thrust for the cell
                            } else {
                                thrustCell[d] = (fg->normal[d])*(cellPressure-101325); // calculate thrust for the cell
                            };
                            thrustTotal[d] = thrustTotal[d] + thrustCell[d]; // summation of total trust along fieldBoundary
                        }
                    }
                }
            }
        }

        // Take across all ranks
        MPI_Reduce(mDotTotal, mDotTotalGlob, bufferSize, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(thrustTotal, thrustTotalGlob, bufferSize, MPI_DOUBLE, MPI_SUM, 0, comm);

        for (PetscInt d = 0; d < dim; d++) {
            if ( tol < abs(thrustTotalGlob[d]) && 1e-2 < abs(mDotTotalGlob[d])) { // avoid nan or dividing to near zero numbers
                IspGlob[d] = thrustTotalGlob[d] / ((mDotTotalGlob[d]) * 9.8);  // calculate specific Impulse
            }
        }

        if (!monitor->log->Initialized()) {
            monitor->log->Initialize(comm);
        }

        if (rank == 0) {
            if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
                if (monitor->name != "") {  // If user passed a name argument then output name
                    monitor->log->Printf("%s ", monitor->name.c_str());
                }
                monitor->log->Printf("RocketMonitor for timestep %04d: time: %-8.4g\n", (int)step, (double)crtime);
                monitor->log->Printf("\tThrust:\t [ %1.7f, %1.7f, %1.7f]\n", thrustTotalGlob[0], thrustTotalGlob[1], thrustTotalGlob[2]);
                monitor->log->Printf("\tIsp:\t [ %1.7f, %1.7f, %1.7f]\n", IspGlob[0], IspGlob[1], IspGlob[2]);
            }
        }

        // cleanup
        VecRestoreArrayRead(faceGeomVec, &faceGeomArray) >> checkError;
        VecRestoreArrayRead(cellGeomVec, &cellGeomArray) >> checkError;
        VecDestroy(&cellGeomVec) >> checkError;
        VecDestroy(&faceGeomVec) >> checkError;
        VecRestoreArrayRead(auxVec, &auxArray) >> checkError;
        VecRestoreArrayRead(solVec, &solArray) >> checkError;


    }
    PetscFunctionReturn(0);

}

#include "registrar.hpp"
REGISTER(ablate::monitors::Monitor, ablate::monitors::RocketMonitor, "Outputs the Thrust and Specific Impulse of a Rocket",
         OPT(std::string, "name", "if provided this name is used to indentify the monitor"),
         ARG(ablate::domain::Region, "region", "the region to apply this solver"),
         ARG(ablate::domain::Region, "fieldBoundary", "the region describing the faces between the boundary and field"),
         ARG(ablate::eos::EOS, "eos", "(ablate::eos::EOS) The EOS describing the flow field at the boundary"),
         OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"),
         OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"));
