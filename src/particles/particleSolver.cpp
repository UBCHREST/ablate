#include "particleSolver.hpp"
#include <petscviewerhdf5.h>
#include <numeric>
#include <utility>
#include "particles/accessors/eulerianAccessor.hpp"
#include "particles/accessors/rhsAccessor.hpp"
#include "particles/accessors/swarmAccessor.hpp"
#include "utilities/mpiUtilities.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::particles::ParticleSolver::ParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::vector<FieldDescription> fields,
                                                  std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                                                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                                                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : Solver(std::move(solverId), std::move(region), std::move(options)),
      fieldsDescriptions(std::move(std::move(fields))),
      processes(std::move(processes)),
      initializer(std::move(initializer)),
      fieldInitialization(std::move(fieldInitialization)),
      exactSolutions(std::move(exactSolutions))

{}
ablate::particles::ParticleSolver::ParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options,
                                                  const std::vector<std::shared_ptr<FieldDescription>> &fields, std::vector<std::shared_ptr<processes::Process>> processes,
                                                  std::shared_ptr<initializers::Initializer> initializer, std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                                                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : ParticleSolver(std::move(solverId), std::move(region), std::move(options), ablate::utilities::VectorUtilities::Copy(fields), std::move(processes), std::move(initializer),
                     std::move(fieldInitialization), std::move(exactSolutions)) {}

ablate::particles::ParticleSolver::~ParticleSolver() {
    if (swarmDm) {
        DMDestroy(&swarmDm) >> checkError;
    }
    if (particleTs) {
        TSDestroy(&particleTs) >> checkError;
    }
}

void ablate::particles::ParticleSolver::Setup() {
    // create and associate the dm
    DMCreate(subDomain->GetComm(), &swarmDm) >> checkError;
    DMSetType(swarmDm, DMSWARM) >> checkError;
    ndims = subDomain->GetDimensions();
    DMSetDimension(swarmDm, ndims) >> checkError;

    /**
     * DMSWARM_PIC is suitable for particle-in-cell methods. Configured as DMSWARM_PIC, the swarm will be aware of, another DM which serves as the background mesh. Fields specific to particle-in-cell
     * methods are registered by default. These include spatial coordinates, a unique identifier, a cell index and an index for the owning rank. The background mesh will (by default) define the
     * spatial decomposition of the points defined in the swarm. DMSWARM_PIC provides support for particle-in-cell operations such as defining initial point coordinates, communicating particles
     * between sub-domains, projecting particle data fields on to the mesh.
     */
    DMSwarmSetType(swarmDm, DMSWARM_PIC) >> checkError;

    // Record the default fields
    std::vector<std::string> coordComponents;
    switch (ndims) {
        case 1:
            coordComponents = {"X"};
            break;
        case 2:
            coordComponents = {"X", "Y"};
            break;
        case 3:
            coordComponents = {"X", "Y", "Z"};
            break;
        default:
            throw std::invalid_argument("Particles ndims must be 1, 2, or 3. " + std::to_string(ndims) + " is not valid.");
    }

    // if the exact solution was provided, register the initial particle location in the field
    if (!exactSolutions.empty()) {
        fieldsDescriptions.emplace_back(ParticleInitialLocation, domain::FieldLocation::AUX, coordComponents);
    }

    // record the automatic field of coordinates
    auto coordField = FieldDescription{ParticleCoordinates, domain::FieldLocation::SOL, coordComponents};
    fieldsDescriptions.push_back(coordField);

    // march over each field and record the field description
    for (auto &fieldsDescription : fieldsDescriptions) {
        RegisterParticleField(fieldsDescription);
    }

    // incase more than one solution field is provided, create a new field to hold them packed together
    {
        std::vector<std::string> solutionFieldComponentNames;

        // March over each solution field and compute the current offset
        for (const auto &field : fields) {
            if (field.location == domain::FieldLocation::SOL) {
                // Create a global list of names for this field
                if (field.components.size() == 1) {
                    solutionFieldComponentNames.push_back(field.name);
                } else {
                    solutionFieldComponentNames.insert(solutionFieldComponentNames.end(), field.components.begin(), field.components.end());
                }
            }
        }

        // Compute the size of the exact solution (each component added up)
        RegisterParticleField(FieldDescription{PackedSolution, domain::FieldLocation::AUX, solutionFieldComponentNames});

        // Update all solution field components with the size of the number of components
        for (auto &field : fields) {
            if (field.location == domain::FieldLocation::SOL) {
                field.dataSize = (PetscInt)solutionFieldComponentNames.size();
            }
        }
        for (auto &field : fieldsMap) {
            if (field.second.location == domain::FieldLocation::SOL) {
                field.second.dataSize = (PetscInt)solutionFieldComponentNames.size();
            }
        }
    }
}

void ablate::particles::ParticleSolver::Initialize() {
    // before setting up the flow finalize the fields
    DMSwarmFinalizeFieldRegister(swarmDm) >> checkError;

    // associate the swarm with the cell dm
    DMSwarmSetCellDM(swarmDm, subDomain->GetDM()) >> checkError;

    // name the particle domain
    PetscObjectSetOptions((PetscObject)swarmDm, petscOptions) >> checkError;
    PetscObjectSetName((PetscObject)swarmDm, GetSolverId().c_str()) >> checkError;
    DMSetFromOptions(swarmDm) >> checkError;

    // initialize the particles
    initializer->Initialize(*subDomain, swarmDm);

    // Setup particle integrator
    TSCreate(subDomain->GetComm(), &particleTs) >> checkError;
    PetscObjectSetOptions((PetscObject)particleTs, petscOptions) >> checkError;
    TSSetApplicationContext(particleTs, this) >> checkError;

    // Link the dm
    TSSetDM(particleTs, swarmDm);
    TSSetProblemType(particleTs, TS_NONLINEAR) >> checkError;
    TSSetExactFinalTime(particleTs, TS_EXACTFINALTIME_MATCHSTEP) >> checkError;
    TSSetMaxSteps(particleTs, 100000000) >> checkError;  // set the max ts to a very large number. This can be overwritten using ts_max_steps options

    // finish ts setup
    TSSetFromOptions(particleTs) >> checkError;

    // set the functions to compute error is provided
    if (!exactSolutions.empty()) {
        StoreInitialParticleLocations();
        TSSetComputeExactError(particleTs, ComputeParticleError) >> checkError;
    }

    // project the initialization field onto each local particle
    for (auto &field : fieldInitialization) {
        this->ProjectFunction(field);
    }

    // Set the start time for TSSolve
    TSSetTime(particleTs, timeInitial) >> checkError;

    // set the particle RHS
    TSSetRHSFunction(particleTs, nullptr, ComputeParticleRHS, this) >> checkError;

    // link the solution with the flowTS
    RegisterPostStep([this](TS flowTs, ablate::solver::Solver &) { this->MacroStepParticles(flowTs); });
}

void ablate::particles::ParticleSolver::RegisterParticleField(const FieldDescription &fieldDescription) {
    // Convert the fieldDescription to a field
    Field field{
        //! The unique name of the particle field
        .name = fieldDescription.name,

        //! The name of the components
        .numberComponents = (PetscInt)fieldDescription.components.size(),

        //! The name of the components
        .components = fieldDescription.components,

        //! The field type (sol or aux)
        .location = fieldDescription.location,

        //! The type of field
        .dataType = fieldDescription.dataType,

        //! The offset in the local array, 0 for aux, computed for sol
        .offset = field.location == domain::FieldLocation::SOL
                      ? std::accumulate(
                            fields.begin(), fields.end(), 0, [&](PetscInt count, const Field &field) { return count + (field.location == domain::FieldLocation::SOL ? field.numberComponents : 0); })
                      : 0,

        //! The size of the component for this data
        .dataSize = (PetscInt)fieldDescription.components.size()};

    // Store the field
    fields.push_back(field);
    fieldsMap.insert({field.name, field});

    // register the field if it is an aux field, sol fields will be added later
    if (field.location == domain::FieldLocation::AUX) {
        // add the value to the field
        DMSwarmRegisterPetscDatatypeField(swarmDm, field.name.c_str(), (PetscInt)field.components.size(), field.dataType) >> checkError;
    }
}

void ablate::particles::ParticleSolver::StoreInitialParticleLocations() {
    // copy over the initial location
    PetscReal *coord;
    PetscReal *initialLocation;
    PetscInt numberParticles;
    DMSwarmGetLocalSize(swarmDm, &numberParticles) >> checkError;
    DMSwarmGetField(swarmDm, DMSwarmPICField_coor, nullptr, nullptr, (void **)&coord) >> checkError;
    DMSwarmGetField(swarmDm, ParticleInitialLocation, nullptr, nullptr, (void **)&initialLocation) >> checkError;

    // copy the raw data
    for (int i = 0; i < numberParticles * ndims; ++i) {
        initialLocation[i] = coord[i];
    }
    DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, nullptr, nullptr, (void **)&coord) >> checkError;
    DMSwarmRestoreField(swarmDm, ParticleInitialLocation, nullptr, nullptr, (void **)&initialLocation) >> checkError;
}

PetscErrorCode ablate::particles::ParticleSolver::ComputeParticleError(TS particleTS, Vec u, Vec errorVec) {
    PetscFunctionBeginUser;
    // get a pointer to this particle class
    ablate::particles::ParticleSolver *particles;
    TSGetApplicationContext(particleTS, (void **)&particles) >> checkError;

    // get the abs time for the particle evaluation, this is the ts relative time plus the time at the start of the particle ts solve
    PetscReal time;
    TSGetTime(particleTS, &time) >> checkError;
    time += particles->timeInitial;

    // Create a vector of the current solution
    Vec exactSolutionVec;
    VecDuplicate(u, &exactSolutionVec) >> checkError;
    PetscScalar *exactSolutionArray;
    VecGetArrayWrite(exactSolutionVec, &exactSolutionArray) >> checkError;

    // Also store the exact location separately
    DMSwarmVectorDefineField(particles->swarmDm, ParticleInitialLocation) >> checkError;
    Vec exactLocationVec;
    DMGetGlobalVector(particles->swarmDm, &exactLocationVec);
    PetscScalar *exactLocationArray;
    VecGetArrayWrite(exactLocationVec, &exactLocationArray) >> checkError;

    // get the initial location array
    const PetscScalar *initialParticleLocationArray;
    const auto &initialParticleLocationField = particles->GetField(ParticleInitialLocation, &initialParticleLocationArray);

    // exact the exact solution from the initial location
    PetscInt np;
    DMSwarmGetLocalSize(particles->swarmDm, &np) >> checkError;
    const PetscInt dim = particles->ndims;

    // March over each field with an exact solution
    for (const auto &exactSolutionFunction : particles->exactSolutions) {
        const auto &exactSolutionField = particles->GetField(exactSolutionFunction->GetName());

        if (exactSolutionField.location != domain::FieldLocation::SOL) {
            throw std::invalid_argument("The exactSolution field (" + exactSolutionField.name + ") must be of type domain::FieldLocation::SOL");
        }

        // extract the petsc function for fast update
        void *functionContext = exactSolutionFunction->GetSolutionField().GetContext();
        ablate::mathFunctions::PetscFunction functionPointer = exactSolutionFunction->GetSolutionField().GetPetscFunction();

        // for each local particle, get the exact location and other variables
        for (PetscInt p = 0; p < np; ++p) {
            // Call the update function
            functionPointer(
                dim, time, initialParticleLocationArray + initialParticleLocationField[p], exactSolutionField.numberComponents, exactSolutionArray + exactSolutionField[p], functionContext) >>
                checkError;
        }

        // Also set the exact solution
        if (exactSolutionField.name == ParticleCoordinates) {
            // for each local particle, get the exact location and other variables
            for (PetscInt p = 0; p < np; ++p) {
                // Call the update function
                functionPointer(dim, time, initialParticleLocationArray + initialParticleLocationField[p], exactSolutionField.numberComponents, exactLocationArray + (p * dim), functionContext) >>
                    checkError;
            }
        }
    }

    VecRestoreArrayWrite(exactSolutionVec, &exactSolutionArray) >> checkError;
    VecRestoreArrayWrite(exactLocationVec, &exactLocationArray) >> checkError;

    // Get all points still in this mesh
    DM flowDM = particles->subDomain->GetDM();
    PetscSF cellSF = nullptr;
    DMLocatePoints(flowDM, exactLocationVec, DM_POINTLOCATION_NONE, &cellSF) >> checkError;
    const PetscSFNode *cells;
    PetscSFGetGraph(cellSF, nullptr, nullptr, nullptr, &cells) >> checkError;

    // compute the difference between exact and u
    VecWAXPY(errorVec, -1, exactSolutionVec, u);

    // get the solution field size to zero the error
    const auto solutionFieldSize = particles->GetField(PackedSolution).dataSize;

    // zero out the error if any particle moves outside the domain
    for (PetscInt p = 0; p < np; ++p) {
        if (cells[p].index == DMLOCATEPOINT_POINT_NOT_FOUND) {
            for (PetscInt c = 0; c < solutionFieldSize; ++c) {
                VecSetValue(errorVec, p * solutionFieldSize + c, 0.0, INSERT_VALUES) >> checkError;
            }
        }
    }
    VecAssemblyBegin(errorVec) >> checkError;
    VecAssemblyEnd(errorVec) >> checkError;

    // restore all the vecs/fields
    PetscSFDestroy(&cellSF) >> checkError;

    // cleanup
    particles->RestoreField(ParticleInitialLocation, &initialParticleLocationArray);

    VecDestroy(&exactSolutionVec) >> checkError;
    DMRestoreGlobalVector(particles->swarmDm, &exactLocationVec) >> checkError;

    PetscFunctionReturn(0);
}

void ablate::particles::ParticleSolver::ProjectFunction(const std::shared_ptr<mathFunctions::FieldFunction> &fieldFunction, PetscReal time) {
    // Get the local number of particles
    PetscInt np;
    DMSwarmGetLocalSize(swarmDm, &np) >> checkError;

    // Get the raw access to position and update field
    PetscInt dim;
    PetscReal *positionData;
    DMSwarmGetField(swarmDm, DMSwarmPICField_coor, &dim, nullptr, (void **)&positionData) >> checkError;

    // Get the field
    PetscReal *fieldData;
    const auto &field = GetField(fieldFunction->GetName(), &fieldData);
    if (field.dataType != PETSC_REAL) {
        throw std::invalid_argument("ProjectFunction only supports PETSC_REAL");
    }

    // extract the petsc function for fast update
    void *functionContext = fieldFunction->GetSolutionField().GetContext();
    ablate::mathFunctions::PetscFunction functionPointer = fieldFunction->GetSolutionField().GetPetscFunction();

    // Iterate over each local particle
    for (PetscInt p = 0; p < np; ++p) {
        // compute the position offset
        const PetscInt positionOffset = p * dim;

        // Compute the field offset
        const PetscInt fieldOffset = field[p];

        // Call the update function
        functionPointer(dim, time, positionData + positionOffset, field.numberComponents, fieldData + fieldOffset, functionContext) >> checkError;
    }
    DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, nullptr, nullptr, (void **)&positionData);
    RestoreField(field, &fieldData);
}
void ablate::particles::ParticleSolver::SwarmMigrate() {
    // current number of local/global particles
    PetscInt numberLocal;
    PetscInt numberGlobal;

    // Get the current size
    DMSwarmGetLocalSize(swarmDm, &numberLocal) >> checkError;
    DMSwarmGetSize(swarmDm, &numberGlobal) >> checkError;

    // Migrate any particles that have moved
    DMSwarmMigrate(swarmDm, PETSC_TRUE) >> checkError;

    // get the new sizes
    PetscInt newNumberLocal;
    PetscInt newNumberGlobal;

    // Get the updated size
    DMSwarmGetLocalSize(swarmDm, &newNumberLocal) >> checkError;
    DMSwarmGetSize(swarmDm, &newNumberGlobal) >> checkError;

    // Check to see if any of the ranks changed size after migration
    PetscMPIInt dmChangedLocal = newNumberGlobal != numberGlobal || newNumberLocal != numberLocal;
    MPI_Comm comm;
    PetscObjectGetComm((PetscObject)particleTs, &comm) >> checkError;
    PetscMPIInt dmChangedAll = PETSC_FALSE;
    MPI_Allreduce(&dmChangedLocal, &dmChangedAll, 1, MPI_INT, MPI_MAX, comm) >> ablate::utilities::MpiUtilities::checkError;
    dmChanged = dmChangedAll > 0;
}

void ablate::particles::ParticleSolver::MacroStepParticles(TS macroTS) {
    // if the dm has changed size (new particles, particles moved between ranks, particles deleted) reset the ts
    if (dmChanged) {
        TSReset(particleTs) >> checkError;
        dmChanged = PETSC_FALSE;
    }

    // Update the coordinates in the solution vector
    CoordinatesToSolutionVector();

    // get the particle time step
    PetscReal dtInitial;
    TSGetTimeStep(particleTs, &dtInitial) >> checkError;

    // Set the max end time based upon the flow end time
    PetscReal time;
    TSGetTime(macroTS, &time) >> checkError;
    TSSetMaxTime(particleTs, time) >> checkError;
    timeFinal = time;

    // get the solution vector as a vector
    Vec solutionVector;
    DMSwarmCreateGlobalVectorFromField(swarmDm, PackedSolution, &solutionVector) >> checkError;

    // take the needed timesteps to get to the flow time
    TSSolve(particleTs, solutionVector) >> checkError;
    timeInitial = timeFinal;

    // get the updated time step, and reset if it has gone down
    PetscReal dtUpdated;
    TSGetTimeStep(particleTs, &dtUpdated) >> checkError;
    if (dtUpdated < dtInitial) {
        TSSetTimeStep(particleTs, dtInitial) >> checkError;
    }

    // put back the vector
    DMSwarmDestroyGlobalVectorFromField(swarmDm, PackedSolution, &solutionVector) >> checkError;

    // Decode the solution vector to coordinates
    CoordinatesFromSolutionVector();

    // Migrate any particles that have moved
    SwarmMigrate();
}
void ablate::particles::ParticleSolver::CoordinatesToSolutionVector() {
    // Get the local number of particles
    PetscInt np;
    DMSwarmGetLocalSize(swarmDm, &np) >> checkError;

    // Get the raw access to position and update field
    PetscInt dim;
    PetscReal *positionData;
    DMSwarmGetField(swarmDm, DMSwarmPICField_coor, &dim, nullptr, (void **)&positionData) >> checkError;

    // Get the field
    PetscReal *fieldData;
    const auto &field = GetField(ParticleCoordinates, &fieldData);

    // Iterate over each local particle
    for (PetscInt p = 0; p < np; ++p) {
        // compute the position offset
        const PetscInt positionOffset = p * dim;

        // Compute the field offset
        const PetscInt fieldOffset = field[p];

        PetscArraycpy(fieldData + fieldOffset, positionData + positionOffset, dim) >> checkError;
    }

    DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, nullptr, nullptr, (void **)&positionData);
    RestoreField(field, &fieldData);
}

void ablate::particles::ParticleSolver::CoordinatesFromSolutionVector() {
    // Get the local number of particles
    PetscInt np;
    DMSwarmGetLocalSize(swarmDm, &np) >> checkError;

    // Get the raw access to position and update field
    PetscInt dim;
    PetscReal *positionData;
    DMSwarmGetField(swarmDm, DMSwarmPICField_coor, &dim, nullptr, (void **)&positionData) >> checkError;

    // Get the field
    PetscReal *fieldData;
    const auto &field = GetField(ParticleCoordinates, &fieldData);

    // Iterate over each local particle
    for (PetscInt p = 0; p < np; ++p) {
        // compute the position offset
        const PetscInt positionOffset = p * dim;

        // Compute the field offset
        const PetscInt fieldOffset = field[p];

        PetscArraycpy(positionData + positionOffset, fieldData + fieldOffset, dim) >> checkError;
    }

    DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, nullptr, nullptr, (void **)&positionData);
    RestoreField(field, &fieldData);
}

PetscErrorCode ablate::particles::ParticleSolver::ComputeParticleRHS(TS ts, PetscReal t, Vec x, Vec f, void *ctx) {
    PetscFunctionBeginUser;

    auto particleSolver = (ablate::particles::ParticleSolver *)ctx;

    // determine if we should cachePointData
    auto cachePointData = particleSolver->processes.size() != 1;

    // Zero out f so that the processes can add do it
    PetscCall(VecZeroEntries(f));

    // Build the needed data structures
    accessors::SwarmAccessor swarmAccessor(cachePointData, particleSolver->swarmDm, particleSolver->fieldsMap, x);
    accessors::RhsAccessor rhsAccessor(cachePointData, particleSolver->fieldsMap, f);
    accessors::EulerianAccessor eulerianAccessor(cachePointData, particleSolver->subDomain, swarmAccessor, t);

    // March over each processes
    try {
        for (auto &processes : particleSolver->processes) {
            processes->ComputeRHS(t, swarmAccessor, rhsAccessor, eulerianAccessor);
        }
    } catch (std::exception &exception) {
        SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_LIB, "%s", exception.what());
    }

    PetscFunctionReturn(0);
}

PetscErrorCode ablate::particles::ParticleSolver::ComputeParticleExactSolution(TS particleTS, Vec exactSolutionVec) {
    PetscFunctionBeginUser;

    // get a pointer to this particle class
    ablate::particles::ParticleSolver *particles;
    TSGetApplicationContext(particleTS, (void **)&particles) >> checkError;

    // get the abs time for the particle evaluation, this is the ts relative time plus the time at the start of the particle ts solve
    PetscReal time;
    TSGetTime(particleTS, &time) >> checkError;
    time += particles->timeInitial;

    // Create a vector of the current solution
    PetscScalar *exactSolutionArray;
    VecGetArrayWrite(exactSolutionVec, &exactSolutionArray) >> checkError;

    // exact the exact solution from the initial location
    PetscInt np;
    DMSwarmGetLocalSize(particles->swarmDm, &np) >> checkError;
    const PetscInt dim = particles->ndims;

    // get the initial location array
    const PetscScalar *initialParticleLocationArray;
    const auto &initialParticleLocationField = particles->GetField(ParticleInitialLocation, &initialParticleLocationArray);

    // March over each field with an exact solution
    for (const auto &exactSolutionFunction : particles->exactSolutions) {
        const auto &exactSolutionField = particles->GetField(exactSolutionFunction->GetName());

        if (exactSolutionField.location != domain::FieldLocation::SOL) {
            throw std::invalid_argument("The exactSolution field (" + exactSolutionField.name + ") must be of type domain::FieldLocation::SOL");
        }

        // extract the petsc function for fast update
        void *functionContext = exactSolutionFunction->GetSolutionField().GetContext();
        ablate::mathFunctions::PetscFunction functionPointer = exactSolutionFunction->GetSolutionField().GetPetscFunction();

        // for each local particle, get the exact location and other variables
        for (PetscInt p = 0; p < np; ++p) {
            // Call the update function
            functionPointer(
                dim, time, initialParticleLocationArray + initialParticleLocationField[p], exactSolutionField.numberComponents, exactSolutionArray + exactSolutionField[p], functionContext) >>
                checkError;
        }
    }

    VecRestoreArrayWrite(exactSolutionVec, &exactSolutionArray) >> checkError;

    // cleanup
    particles->RestoreField(ParticleInitialLocation, &initialParticleLocationArray);
    PetscFunctionReturn(0);
}

static PetscErrorCode DMSequenceViewTimeHDF5(DM dm, PetscViewer viewer) {
    Vec stamp;
    PetscMPIInt rank;

    PetscFunctionBegin;
    // get the seqnum and value from the dm
    PetscInt seqnum;
    PetscReal value;
    PetscCall(DMGetOutputSequenceNumber(dm, &seqnum, &value));

    if (seqnum < 0) {
        PetscFunctionReturn(0);
    }
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)viewer), &rank));
    PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)viewer), rank ? 0 : 1, 1, &stamp));
    PetscCall(VecSetBlockSize(stamp, 1));
    PetscCall(PetscObjectSetName((PetscObject)stamp, "time"));
    if (!rank) {
        PetscCall(VecSetValue(stamp, 0, value, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(stamp));
    PetscCall(VecAssemblyEnd(stamp));
    PetscCall(PetscViewerHDF5PushGroup(viewer, "/"));
    PetscCall(PetscViewerHDF5SetTimestep(viewer, seqnum));
    PetscCall(VecView(stamp, viewer));
    PetscCall(PetscViewerHDF5PopGroup(viewer));
    PetscCall(VecDestroy(&stamp));
    PetscFunctionReturn(0);
}

void ablate::particles::ParticleSolver::Save(PetscViewer viewer, PetscInt steps, PetscReal time) {
    PetscFunctionBeginUser;
    DMSetOutputSequenceNumber(GetParticleDM(), steps, time) >> checkError;
    Vec particleVector;

    // if this is an hdf5Viewer
    PetscBool ishdf5;
    PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5) >> checkError;
    if (ishdf5) {
        PetscBool isInTimestepping;
        PetscViewerHDF5IsTimestepping(viewer, &isInTimestepping) >> checkError;
        if (!isInTimestepping) {
            PetscViewerHDF5PushTimestepping(viewer) >> checkError;
        }
    }

    // output the default coordinate field
    DMSwarmCreateGlobalVectorFromField(GetParticleDM(), DMSwarmPICField_coor, &particleVector) >> checkError;
    PetscObjectSetName((PetscObject)particleVector, DMSwarmPICField_coor) >> checkError;
    VecView(particleVector, viewer) >> checkError;
    DMSwarmDestroyGlobalVectorFromField(GetParticleDM(), DMSwarmPICField_coor, &particleVector) >> checkError;

    // output all the fields
    for (auto const &field : fields) {
        if (field.dataType == PETSC_REAL && field.location == domain::FieldLocation::AUX) {
            DMSwarmCreateGlobalVectorFromField(GetParticleDM(), field.name.c_str(), &particleVector) >> checkError;
            PetscObjectSetName((PetscObject)particleVector, field.name.c_str()) >> checkError;
            VecView(particleVector, viewer) >> checkError;

            // write the field components to the file if hdf5
            if (ishdf5 && field.numberComponents > 1) {
                PetscViewerHDF5PushGroup(viewer, "/particle_fields") >> checkError;

                for (std::size_t c = 0; c < field.components.size(); c++) {
                    std::string componentNameLabel = "componentName" + std::to_string(c);
                    PetscViewerHDF5WriteObjectAttribute(viewer, (PetscObject)particleVector, componentNameLabel.c_str(), PETSC_STRING, field.components[c].c_str()) >> checkError;
                }

                PetscViewerHDF5PopGroup(viewer) >> checkError;
            }

            DMSwarmDestroyGlobalVectorFromField(GetParticleDM(), field.name.c_str(), &particleVector) >> checkError;
        }
    }

    // Get the particle info
    int rank;
    MPI_Comm_rank(PetscObjectComm((PetscObject)GetParticleDM()), &rank) >> ablate::utilities::MpiUtilities::checkError;

    // get the local number of particles
    PetscInt globalSize;
    DMSwarmGetSize(GetParticleDM(), &globalSize) >> ablate::utilities::MpiUtilities::checkError;

    // record the number of particles per rank
    Vec particleCountVec;
    VecCreateMPI(PetscObjectComm((PetscObject)GetParticleDM()), PETSC_DECIDE, 1, &particleCountVec) >> checkError;
    PetscObjectSetName((PetscObject)particleCountVec, "particleCount") >> checkError;
    VecSetValue(particleCountVec, 0, globalSize, INSERT_VALUES) >> checkError;
    VecAssemblyBegin(particleCountVec) >> checkError;
    VecAssemblyEnd(particleCountVec) >> checkError;
    VecView(particleCountVec, viewer);
    VecDestroy(&particleCountVec) >> checkError;

    if (ishdf5) {
        DMSequenceViewTimeHDF5(GetParticleDM(), viewer) >> checkError;
    }
    PetscFunctionReturnVoid();
}
void ablate::particles::ParticleSolver::Restore(PetscViewer viewer, PetscInt sequenceNumber, PetscReal time) {
    DMSetOutputSequenceNumber(GetParticleDM(), sequenceNumber, time) >> checkError;

    // Update the ts with the current values
    TSSetTime(particleTs, time) >> checkError;
    timeInitial = time;

    // There is not a hdf5 specific swarm vec load, so that needs to be in this code
    PetscBool ishdf5;
    PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERHDF5, &ishdf5) >> checkError;
    if (ishdf5) {
        PetscViewerHDF5PushTimestepping(viewer) >> checkError;
        PetscViewerHDF5SetTimestep(viewer, sequenceNumber) >> checkError;
    }

    // load in the global particle size
    Vec particleCountVec;
    VecCreateSeq(PETSC_COMM_SELF, 1, &particleCountVec) >> checkError;
    PetscObjectSetName((PetscObject)particleCountVec, "particleCount") >> checkError;
    VecLoad(particleCountVec, viewer) >> checkError;

    PetscReal globalSize;
    PetscInt index[1] = {0};
    VecGetValues(particleCountVec, 1, index, &globalSize) >> checkError;
    VecDestroy(&particleCountVec) >> checkError;

    // Get the particle mpi, info
    int rank, size;
    MPI_Comm_rank(PetscObjectComm((PetscObject)GetParticleDM()), &rank) >> ablate::utilities::MpiUtilities::checkError;
    MPI_Comm_size(PetscObjectComm((PetscObject)GetParticleDM()), &size) >> ablate::utilities::MpiUtilities::checkError;

    // distribute the number of particles across all ranks
    PetscInt localSize = ((PetscInt)globalSize) / size;

    // Use the first rank to hold any left over
    if (rank == 0) {
        localSize = globalSize - (localSize * (size - 1));
    }

    // Set the local swarm size
    DMSwarmSetLocalSizes(GetParticleDM(), localSize, 0) >> checkError;

    // Move in the hdf5 to the right group
    if (ishdf5) {
        PetscViewerHDF5PushGroup(viewer, "/particle_fields") >> checkError;
        PetscViewerHDF5SetTimestep(viewer, sequenceNumber) >> checkError;
    }

    {  // restore the default coordinate field
        Vec particleVector;
        Vec particleVectorLoad;
        DMSwarmCreateGlobalVectorFromField(swarmDm, DMSwarmPICField_coor, &particleVector) >> checkError;

        // A copy of this vector is needed, because vec load breaks the memory linkage between the swarm and vec
        VecDuplicate(particleVector, &particleVectorLoad) >> checkError;

        // Load the vector
        PetscObjectSetName((PetscObject)particleVectorLoad, DMSwarmPICField_coor) >> checkError;
        VecLoad(particleVectorLoad, viewer) >> checkError;

        // Copy the data over
        VecCopy(particleVectorLoad, particleVector) >> checkError;

        DMSwarmDestroyGlobalVectorFromField(swarmDm, DMSwarmPICField_coor, &particleVector) >> checkError;
        VecDestroy(&particleVectorLoad) >> checkError;
    }

    // restore the aux vectors
    for (auto const &field : fields) {
        if (field.dataType == PETSC_REAL && field.location == domain::FieldLocation::AUX) {
            Vec particleVector;
            Vec particleVectorLoad;
            DMSwarmCreateGlobalVectorFromField(swarmDm, field.name.c_str(), &particleVector) >> checkError;

            // A copy of this vector is needed, because vec load breaks the memory linkage between the swarm and vec
            VecDuplicate(particleVector, &particleVectorLoad) >> checkError;

            // Load the vector
            PetscObjectSetName((PetscObject)particleVectorLoad, field.name.c_str()) >> checkError;
            VecLoad(particleVectorLoad, viewer) >> checkError;

            // Copy the data over
            VecCopy(particleVectorLoad, particleVector) >> checkError;

            DMSwarmDestroyGlobalVectorFromField(swarmDm, field.name.c_str(), &particleVector) >> checkError;
            VecDestroy(&particleVectorLoad) >> checkError;
        }
    }

    if (ishdf5) {
        PetscViewerHDF5PopGroup(viewer) >> checkError;
        PetscViewerHDF5PopTimestepping(viewer) >> checkError;
    }

    // Migrate the particle to the correct rank for the dmPlex
    DMSwarmMigrate(swarmDm, PETSC_TRUE) >> checkError;
    dmChanged = true;
}

#include "registrar.hpp"
REGISTER(ablate::solver::Solver, ablate::particles::ParticleSolver, "Lagrangian particle solver", ARG(std::string, "id", "the name of the particle solver"),
         OPT(ablate::domain::Region, "region", "the region to apply this solver.  Default is entire domain"),
         OPT(ablate::parameters::Parameters, "options", "the options passed to PETSC for the flow"),
         OPT(std::vector<ablate::particles::FieldDescription>, "fields", "any additional fields beside coordinates"),
         ARG(std::vector<ablate::particles::processes::Process>, "processes", "the processes used to describe the particle source terms"),
         ARG(ablate::particles::initializers::Initializer, "initializer", "the initial particle setup methods"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "fieldInitialization", "the initial particle fields values"),
         OPT(std::vector<ablate::mathFunctions::FieldFunction>, "exactSolutions", "particle fields (SOL) exact solutions"));