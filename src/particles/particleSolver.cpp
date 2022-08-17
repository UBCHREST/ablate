#include "particleSolver.hpp"
#include <numeric>
#include "particles/accessors/eulerianAccessor.hpp"
#include "particles/accessors/rhsAccessor.hpp"
#include "particles/accessors/swarmAccessor.hpp"
#include "utilities/mpiError.hpp"

ablate::particles::ParticleSolver::ParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::vector<FieldDescription> fields,
                                                  std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                                                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization,
                                                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> exactSolutions)
    : Solver(solverId, region, options),
      fieldsDescriptions(fields),
      processes(processes),
      initializer(initializer),
      fieldInitialization(fieldInitialization),
      exactSolutions(exactSolutions)

{}
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
        fieldsDescriptions.push_back(FieldDescription{.name = ParticleInitialLocation, .components = coordComponents, .type = domain::FieldLocation::AUX, .dataType = PETSC_REAL});
    }

    // record the automatic field of coordinates
    auto coordField = FieldDescription{.name = ParticleCoordinates, .components = coordComponents, .type = domain::FieldLocation::SOL, .dataType = PETSC_REAL};
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
            if (field.type == domain::FieldLocation::SOL) {
                // Create a global list of names for this field
                if (field.components.size() == 1) {
                    solutionFieldComponentNames.push_back(field.name);
                } else {
                    solutionFieldComponentNames.insert(solutionFieldComponentNames.end(), field.components.begin(), field.components.end());
                }
            }
        }

        // Compute the size of the exact solution (each component added up)
        RegisterParticleField(FieldDescription{.name = PackedSolution, .components = solutionFieldComponentNames, .type = domain::FieldLocation::AUX, .dataType = PETSC_REAL});

        // Update all solution field components with the size of the number of components
        for (auto &field : fields) {
            if (field.type == domain::FieldLocation::SOL) {
                field.dataSize = solutionFieldComponentNames.size();
            }
        }
        for (auto &field : fieldsMap) {
            if (field.second.type == domain::FieldLocation::SOL) {
                field.second.dataSize = solutionFieldComponentNames.size();
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
    TSSetMaxSteps(particleTs, 100000000) >> checkError;  // set the max ts to a very large number. This can be over written using ts_max_steps options

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
    TSSetRHSFunction(particleTs, NULL, ComputeParticleRHS, this) >> checkError;

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
        .type = fieldDescription.type,

        //! The type of field
        .dataType = fieldDescription.dataType,

        //! The offset in the local array, 0 for aux, computed for sol
        .offset =
            field.type == domain::FieldLocation::SOL
                ? std::accumulate(fields.begin(), fields.end(), 0, [&](PetscInt count, const Field &field) { return count + (field.type == domain::FieldLocation::SOL ? field.numberComponents : 0); })
                : 0,

        //! The size of the component for this data
        .dataSize = (PetscInt)fieldDescription.components.size()};

    // Store the field
    fields.push_back(field);
    fieldsMap.insert({field.name, field});

    // register the field if it is an aux field, sol fields will be added later
    if (field.type == domain::FieldLocation::AUX) {
        // add the value to the field
        DMSwarmRegisterPetscDatatypeField(swarmDm, field.name.c_str(), field.components.size(), field.dataType) >> checkError;
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

        if (exactSolutionField.type != domain::FieldLocation::SOL) {
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
    PetscSF cellSF = NULL;
    DMLocatePoints(flowDM, exactLocationVec, DM_POINTLOCATION_NONE, &cellSF) >> checkError;
    const PetscSFNode *cells;
    PetscSFGetGraph(cellSF, NULL, NULL, NULL, &cells) >> checkError;

    // compute the difference between exact and u
    VecWAXPY(errorVec, -1, exactSolutionVec, u);

    // get the solution field size to zero the the error
    const auto solutionFieldSize = particles->GetField(PackedSolution).dataSize;

    // zero out the error if any particle moves outside of the domain
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
    MPI_Allreduce(&dmChangedLocal, &dmChangedAll, 1, MPIU_INT, MPIU_MAX, comm) >> checkMpiError;
    dmChanged = dmChangedAll == PETSC_TRUE;
}

void ablate::particles::ParticleSolver::MacroStepParticles(TS macroTS) {
    PetscReal time;

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

        if (exactSolutionField.type != domain::FieldLocation::SOL) {
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
