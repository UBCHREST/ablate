#include "particleSolver.hpp"
ablate::particles::ParticleSolver::ParticleSolver(std::string solverId, std::shared_ptr<domain::Region> region, std::shared_ptr<parameters::Parameters> options, std::vector<ParticleField> fields,
                                                  std::vector<std::shared_ptr<processes::Process>> processes, std::shared_ptr<initializers::Initializer> initializer,
                                                  std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldInitialization, std::shared_ptr<mathFunctions::MathFunction> exactSolution)
    : Solver(solverId, region, options),
      fields(fields),
      processes(processes),
      initializer(initializer),
      fieldInitialization(fieldInitialization)

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
    if (exactSolution) {
        fields.push_back(ParticleField{.name = ParticleInitialLocation, .components = coordComponents, .type = domain::FieldLocation::AUX, .dataType = PETSC_REAL});
    }

    // register each field
    for (auto &field : fields) {
        RegisterParticleField(field);
    }

    // add back in the default fields
    auto coordField = ParticleField{.name = DMSwarmPICField_coor, .components = coordComponents, .type = domain::FieldLocation::SOL, .dataType = PETSC_REAL};
    solutionFields.push_back(coordField);
    solutionFields.push_back(coordField);
    fields.push_back(ParticleField{.name = DMSwarmField_pid, .type = domain::FieldLocation::AUX, .dataType = PETSC_INT64});

    // if more than one solution field is provided, create a new field to hold them packed together
    if (solutionFields.size() > 1) {
        auto packedSolutionComponentSize = 0;

        for (const auto &solution : solutionFields) {
            packedSolutionComponentSize += solution.components.size();
        }

        // Compute the size of the exact solution (each component added up)
        RegisterParticleField(
            ParticleField{.name = PackedSolution, .components = std::vector<std::string>(packedSolutionComponentSize, "_"), .type = domain::FieldLocation::AUX, .dataType = PETSC_REAL});
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
    if (exactSolution) {
        StoreInitialParticleLocations();
        TSSetComputeExactError(particleTs, ComputeParticleError) >> checkError;
    }

    // project the initialization field onto each local particle
    for (auto &field : fieldInitialization) {
        this->ProjectFunction(field);
    }
}

void ablate::particles::ParticleSolver::RegisterParticleField(const ablate::particles::ParticleField &field) {
    // add the value to the field
    DMSwarmRegisterPetscDatatypeField(swarmDm, field.name.c_str(), field.components.size(), field.dataType) >> checkError;

    // store the field if it is a solution field
    if (field.type == domain::FieldLocation::SOL) {
        solutionFields.push_back(field);
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
PetscErrorCode ablate::particles::ParticleSolver::ComputeParticleError(TS particleTS, Vec u,  Vec errorVec) {
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

    // exact the exact solution from the initial location
    PetscInt np;
    DMSwarmGetLocalSize(particles->swarmDm, &np) >> checkError;
    const PetscInt dim = particles->ndims;

    // Calculate the size of solution field
    // TODO: Change this to set individual/multiple fields in the exact solution
    PetscInt solutionFieldSize = 0;
    for (const auto &field : particles->solutionFields) {
        solutionFieldSize += field.components.size();
    }

    // get the initial location array
    const PetscScalar *initialParticleLocationArray;
    DMSwarmGetField(particles->swarmDm, ParticleInitialLocation, NULL, NULL, (void **)&initialParticleLocationArray) >> checkError;

    // extract the petsc function for fast update
    void *functionContext = particles->exactSolution->GetContext();
    ablate::mathFunctions::PetscFunction functionPointer = particles->exactSolution->GetPetscFunction();

    // for each local particle, get the exact location and other variables
    for (PetscInt p = 0; p < np; ++p) {
        // compute the array offset
        const PetscInt initialPositionOffset = p * dim;
        const PetscInt fieldOffset = p * solutionFieldSize;

        // Call the update function
        functionPointer(dim, time, initialParticleLocationArray + initialPositionOffset, solutionFieldSize, exactSolutionArray + fieldOffset, functionContext) >> checkError;

        // copy over the first dim to the exact solution array
        for (PetscInt d = 0; d < dim; ++d) {
            exactLocationArray[initialPositionOffset + d] = exactSolutionArray[fieldOffset + d];
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
    DMSwarmRestoreField(particles->swarmDm, ParticleInitialLocation, NULL, NULL, (void **)&initialParticleLocationArray) >> checkError;
    VecDestroy(&exactSolutionVec) >> checkError;
    DMRestoreGlobalVector(particles->swarmDm, &exactLocationVec) >> checkError;

    PetscFunctionReturn(0);
}
void ablate::particles::ParticleSolver::ProjectFunction(const std::shared_ptr<mathFunctions::FieldFunction>& fieldFunction) {
    // Get the local number of particles
    PetscInt np;
    DMSwarmGetLocalSize(swarmDm, &np) >> checkError;

    // Get the raw access to position and update field
    PetscInt dim;
    PetscReal *positionData;
    DMSwarmGetField(swarmDm, DMSwarmPICField_coor, &dim, NULL, (void **)&positionData) >> checkError;

    // Get the field name
    const auto& fieldName = fieldFunction->GetName();

    PetscInt fieldComponents;
    PetscDataType fieldType;
    PetscReal *fieldData;
    DMSwarmGetField(swarmDm, fieldName.c_str(), &fieldComponents, &fieldType, (void **)&fieldData) >> checkError;

    if (fieldType != PETSC_REAL) {
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
        const PetscInt fieldOffset = p * fieldComponents;

        // Call the update function
        functionPointer(dim, 0.0, positionData + positionOffset, fieldComponents, fieldData + fieldOffset, functionContext) >> checkError;
    }
    DMSwarmRestoreField(swarmDm, DMSwarmPICField_coor, NULL, NULL, (void **)&positionData);
    DMSwarmRestoreField(swarmDm, fieldName.c_str(), NULL, NULL, (void **)&fieldData);

}
