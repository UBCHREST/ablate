#include "rocketMonitor.hpp"
#include <utilities/mpiError.hpp>
#include <utilities/petscError.hpp>
#include <utility>
#include "io/interval/fixedInterval.hpp"
#include "monitor.hpp"
#include "monitors/logs/log.hpp"
#include "monitors/logs/stdOut.hpp"

ablate::monitors::RocketMonitor::RocketMonitor(const std::string nameIn, std::shared_ptr<domain::Region> regionIn, std::shared_ptr<domain::Region> fieldBoundaryIn, std::shared_ptr<eos::EOS> eosIn,
                                               const std::shared_ptr<logs::Log>& logIn, const std::shared_ptr<io::interval::Interval>& intervalIn, double referencePressureIn)
    : name(nameIn),
      region(std::move(regionIn)),
      fieldBoundary(std::move(fieldBoundaryIn)),
      eos(std::move(eosIn)),
      log(logIn ? logIn : std::make_shared<logs::StdOut>()),
      interval(intervalIn ? intervalIn : std::make_shared<io::interval::FixedInterval>()),
      referencePressure(referencePressureIn ? referencePressureIn : 101325) {}

void ablate::monitors::RocketMonitor::Register(std::shared_ptr<solver::Solver> solver) {
    Monitor::Register(solver);
    auto comm = GetSolver()->GetSubDomain().GetComm();                                                                                       // The communicator in which the reduction takes place.
    computePressure = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::Pressure, GetSolver()->GetSubDomain().GetFields());          // get decode state function/context
    computeSpeedOfSound = eos->GetThermodynamicFunction(eos::ThermodynamicProperty::SpeedOfSound, GetSolver()->GetSubDomain().GetFields());  // get decode state function/context
    if (!log->Initialized()) {
        log->Initialize(comm);
    }
}

PetscErrorCode ablate::monitors::RocketMonitor::OutputRocket(TS ts, PetscInt step, PetscReal crtime, Vec u, void* ctx) {
    PetscFunctionBeginUser;
    auto monitor = (ablate::monitors::RocketMonitor*)ctx;

    if (monitor->interval->Check(PetscObjectComm((PetscObject)ts), step, crtime)) {
        auto dm = monitor->GetSolver()->GetSubDomain().GetDM();      // get the dm
        auto solDM = monitor->GetSolver()->GetSubDomain().GetDM();   // get the sol dm
        auto comm = monitor->GetSolver()->GetSubDomain().GetComm();  // The communicator in which the reduction takes place.

        PetscInt dim;
        DMGetDimension(dm, &dim);           // get the dimensions of the dm
        PetscMPIInt bufferSize;             // initialize buffer size
        PetscMPIIntCast(dim, &bufferSize);  // define the number of elements in buffer
        // check to see if there is a ghost label
        DMLabel ghostLabel;
        DMGetLabel(dm, "ghost", &ghostLabel) >> checkError;

        const auto& fieldEuler = monitor->GetSolver()->GetSubDomain().GetField("euler");  // get the euler field
        PetscReal* cellEuler;
        PetscReal* conservedValues;
        PetscReal cellPressure;
        PetscReal cellVelocity;
        PetscReal cellSpeedOfSound;

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
        PetscReal mDotCell[3] = {0, 0, 0};
        PetscReal mDotTotal[3] = {0, 0, 0};
        PetscReal thrustCell[3] = {0, 0, 0};
        PetscReal thrustTotal[3] = {0, 0, 0};

        // initialize global vectors (receive_buffer)
        PetscReal mDotTotalGlob[3] = {0, 0, 0};
        PetscReal thrustTotalGlob[3] = {0, 0, 0};
        PetscReal IspGlob[3] = {0, 0, 0};
        PetscReal minFieldsGlob[3] = {0, 0, 0};
        PetscReal maxFieldsGlob[3] = {0, 0, 0};
        PetscReal sumFieldsGlob[3] = {0, 0, 0};

        // initialize field values to calculate min, max, and sum for [cell pressure , machNumber]

        // initialize field value arrays min, max, sum [pressure, machNumber]
        PetscReal minFields[2] = {PETSC_MAX_REAL, PETSC_MAX_REAL};
        PetscReal maxFields[2] = {PETSC_MIN_REAL, PETSC_MIN_REAL};
        PetscReal sumFields[2] = {0, 0};
        PetscReal machNumber;
        PetscReal numCells[1] = {0};
        PetscReal numCellsGlob[1];

        // find all faces
        PetscInt fStart, fEnd;
        DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd) >> checkError;

        /** Get the current rank associated with this process */
        PetscMPIInt rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //!< Get the origin rank of the current process. The particle belongs to this rank. The rank only needs to be read once.

        // need to look for faces at specified fieldBoundary then find cells bordering those faces which are in specified region
        for (PetscInt face = fStart; face < fEnd; ++face) {                            // Iterate through all faces to check if in fieldBoundary
            if (ablate::domain::Region::InRegion(monitor->fieldBoundary, dm, face)) {  // Check if each face is in fieldBoundary
                PetscFVFaceGeom* fg;
                DMPlexPointLocalRead(faceDM, face, faceGeomArray, &fg) >> checkError;  // read face geometry for face

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

                        DMPlexPointLocalRead(solDM, neighborCells[n], solArray, &conservedValues);                                  // Retrieve conserved values from cell
                        monitor->computePressure.function(conservedValues, &cellPressure, monitor->computePressure.context.get());  // Retrieve pressure from cell
                        DMPlexPointLocalFieldRead(solDM, neighborCells[n], fieldEuler.id, solArray, &cellEuler);                    // retrieve euler field for density, density*velocity

                        for (PetscInt d = 0; d < dim; d++) {
                            mDotCell[d] = fg->normal[d] * cellEuler[finiteVolume::CompressibleFlowFields::RHOU + d];  // calculate mass flow rate for the cell
                            mDotTotal[d] = mDotTotal[d] + mDotCell[d];                                                // summation of total mass flow rate along fieldBoundary
                            if (abs(mDotCell[d]) > tol) {
                                thrustCell[d] = (mDotCell[d]) * ((cellEuler[finiteVolume::CompressibleFlowFields::RHOU + d]) / (cellEuler[finiteVolume::CompressibleFlowFields::RHO])) +
                                                (fg->normal[d]) * (cellPressure - monitor->referencePressure);  // calculate thrust for the cell
                            } else {
                                thrustCell[d] = (fg->normal[d]) * (cellPressure - 101325);  // calculate thrust for the cell
                            };
                            thrustTotal[d] = thrustTotal[d] + thrustCell[d];  // summation of total trust along fieldBoundary
                        }
                        // update min and max cell pressure
                        if (cellPressure < minFields[0]) {
                            minFields[0] = cellPressure;
                        }
                        if (cellPressure > maxFields[0]) {
                            maxFields[0] = cellPressure;
                        }
                        // update cell pressure sum for mean calculation
                        sumFields[0] += cellPressure;

                        // sum the square of the velocity in each direction, then take the square root to calculate the velocity cell velocity
                        for (PetscInt d = 0; d < dim; d++) {
                            cellVelocity += PetscSqr((cellEuler[finiteVolume::CompressibleFlowFields::RHOU + d]) / (cellEuler[finiteVolume::CompressibleFlowFields::RHO]));
                        }
                        cellVelocity = PetscSqrtReal(cellVelocity);
                        monitor->computeSpeedOfSound.function(conservedValues, &cellSpeedOfSound, monitor->computeSpeedOfSound.context.get());
                        machNumber = cellVelocity / cellSpeedOfSound;

                        // update min and max mach number
                        if (machNumber < minFields[1]) {
                            minFields[1] = machNumber;
                        }
                        if (machNumber > maxFields[1]) {
                            maxFields[1] = machNumber;
                        }
                        // update mach number sum for mean calculation
                        sumFields[1] += machNumber;
                        numCells[0]++;  // Track number of cells on face in region of interest
                    }
                }
            }
        }

        // Take across all ranks
        MPI_Reduce(minFields, minFieldsGlob, 2, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(maxFields, maxFieldsGlob, 2, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(sumFields, sumFieldsGlob, 2, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(numCells, numCellsGlob, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(mDotTotal, mDotTotalGlob, bufferSize, MPI_DOUBLE, MPI_SUM, 0, comm);
        MPI_Reduce(thrustTotal, thrustTotalGlob, bufferSize, MPI_DOUBLE, MPI_SUM, 0, comm);

        // compile fields into vector [min max mean]
        PetscReal pressureField[3] = {minFieldsGlob[0], maxFieldsGlob[0], sumFieldsGlob[0] / numCellsGlob[0]};
        PetscReal machNumberField[3] = {minFieldsGlob[1], maxFieldsGlob[1], sumFieldsGlob[1] / numCellsGlob[0]};

        for (PetscInt d = 0; d < dim; d++) {
            if (tol < abs(thrustTotalGlob[d]) && 1e-2 < abs(mDotTotalGlob[d])) {  // avoid nan or dividing to near zero numbers
                IspGlob[d] = thrustTotalGlob[d] / ((mDotTotalGlob[d]) * 9.8);     // calculate specific Impulse
            }
        }

        // print output
        if (rank == 0) {
            if (monitor->name != "") {  // If user passed a name argument then output name
                monitor->log->Printf("%s ", monitor->name.c_str());
            }
            monitor->log->Printf("RocketMonitor for timestep %04d: time: %-8.4g\n", (int)step, (double)crtime);
            monitor->log->Printf("\tOutputs: [x y z]\n");
            monitor->log->Printf("\tThrust:\t\t [ %1.7f, %1.7f, %1.7f ]\n", thrustTotalGlob[0], thrustTotalGlob[1], thrustTotalGlob[2]);
            monitor->log->Printf("\tIsp:\t\t [ %1.7f, %1.7f, %1.7f ]\n", IspGlob[0], IspGlob[1], IspGlob[2]);
            monitor->log->Printf("\tmDot:\t\t [ %1.7f, %1.7f, %1.7f ]\n", mDotTotalGlob[0], mDotTotalGlob[1], mDotTotalGlob[2]);
            monitor->log->Printf("\tAdditional Fields: [min max mean]\n");
            monitor->log->Printf("\tPressure:\t [ %1.7f, %1.7f, %1.7f ]\n", pressureField[0], pressureField[1], pressureField[2]);
            monitor->log->Printf("\tmachNumber:\t [ %1.7f, %1.7f, %1.7f ]\n", machNumberField[0], machNumberField[1], machNumberField[2]);
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
         OPT(std::string, "name", "if provided this name is used to indentify the monitor"), ARG(ablate::domain::Region, "region", "the region to apply this solver"),
         ARG(ablate::domain::Region, "fieldBoundary", "the region describing the faces between the boundary and field"),
         ARG(ablate::eos::EOS, "eos", "(ablate::eos::EOS) The EOS describing the flow field at the boundary"), OPT(ablate::monitors::logs::Log, "log", "where to record log (default is stdout)"),
         OPT(ablate::io::interval::Interval, "interval", "report interval object, defaults to every"), OPT(int, "referencePressure", "the ambient air pressure (default is 101325 Pa)"));
