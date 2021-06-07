#include "particles.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::particles::Particles::Particles(std::string name, int ndims, std::shared_ptr<particles::initializers::Initializer> initializer, std::shared_ptr<mathFunctions::MathFunction> exactSolution, std::shared_ptr<parameters::Parameters> options)
    : name(name), ndims(ndims), timeInitial(0.0), timeFinal(0.0), dmChanged(false), initializer(initializer), exactSolution(exactSolution), petscOptions(NULL){

    // create and associate the dm
    DMCreate(PETSC_COMM_WORLD, &dm) >> checkError;
    DMSetType(dm, DMSWARM)  >> checkError;
    DMSetDimension(dm, ndims) >> checkError;
    DMSwarmSetType(dm, DMSWARM_PIC) >> checkError;

    // Record the default fields
    particleFieldDescriptors.push_back(particles::ParticleFieldDescriptor{
        .fieldName = DMSwarmPICField_coor,
        .components = ndims,
        .type = PETSC_DOUBLE
    });
    particleFieldDescriptors.push_back(particles::ParticleFieldDescriptor{
        .fieldName = DMSwarmField_pid,
        .components = 1,
        .type = PETSC_INT64
    });

    // if the exact solution was provided, register the initial particle location in the field
    if(exactSolution){
        RegisterField(ParticleFieldDescriptor{
            .fieldName = ParticleInitialLocation,
            .components = ndims,
            .type = PETSC_REAL,
        });
    }

    // Set the options
    if (options) {
        PetscOptionsCreate(&petscOptions) >> checkError;
        options->Fill(petscOptions);
    }
}

void ablate::particles::Particles::InitializeFlow(std::shared_ptr<flow::Flow> flow) {
    // before setting up the flow finalize the fields
    DMSwarmFinalizeFieldRegister(dm) >> checkError;

    // associate the swarm with the cell dm
    DMSwarmSetCellDM(dm, flow->GetDM()) >> checkError;

    // Store the values in the particles from the ts and flow
    flowFinal = flow->GetSolutionVector();
    VecDuplicate(flowFinal, &(flowInitial)) >> checkError;
    VecCopy(flow->GetSolutionVector(), flowInitial) >> checkError;
    flowVelocityFieldIndex = flow->GetFieldId("velocity").value();

    // name the particle domain
    auto namePrefix = name + "_";
    PetscObjectSetOptions((PetscObject)dm, petscOptions) >> checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> checkError;
    DMSetFromOptions(dm) >> checkError;

    // initialize the particles
    initializer->Initialize(*flow, dm);

    // Setup particle position integrator
    TSCreate(PetscObjectComm((PetscObject)flow->GetDM()), &particleTs) >> checkError;
    PetscObjectSetOptions((PetscObject)particleTs, petscOptions) >> checkError;
    TSSetApplicationContext(particleTs, this) >> checkError;

    // Link thw dm
    TSSetDM(particleTs, dm);
    TSSetProblemType(particleTs, TS_NONLINEAR) >> checkError;
    TSSetExactFinalTime(particleTs, TS_EXACTFINALTIME_MATCHSTEP) >> checkError;
    TSSetMaxSteps(particleTs, 100000000) >> checkError; // set the max ts to a very large number. This can be over written using ts_max_steps options

    // finish ts setup
    TSSetFromOptions(particleTs) >> checkError;

    // set the functions to compute error is provided
    if(exactSolution){
        StoreInitialParticleLocations();
        TSSetComputeExactError(particleTs, ComputeParticleError) >> checkError;
    }
}

ablate::particles::Particles::~Particles() {
    if (dm){
        DMDestroy(&dm) >> checkError;
    }
    if (particleTs) {
        TSDestroy(&particleTs) >> checkError;
    }
    if(flowInitial){
        VecDestroy(&flowInitial) >> checkError;
    }
    if (petscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck(name, &petscOptions);
    }
}

void ablate::particles::Particles::RegisterField(ParticleFieldDescriptor fieldDescriptor) {

    // add the value to the field
     DMSwarmRegisterPetscDatatypeField(dm, fieldDescriptor.fieldName.c_str(), fieldDescriptor.components, fieldDescriptor.type) >> checkError;

     // store the field
     particleFieldDescriptors.push_back(fieldDescriptor);
}

void ablate::particles::Particles::StoreInitialParticleLocations(){
    // copy over the initial location
    PetscReal *coord;
    PetscReal *initialLocation;
    PetscInt numberParticles;
    DMSwarmGetLocalSize(dm, &numberParticles) >> checkError;
    DMSwarmGetField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord) >> checkError;
    DMSwarmGetField(dm, ParticleInitialLocation, NULL, NULL, (void **)&initialLocation) >> checkError;

    // copy the raw data
    for (int i = 0; i < numberParticles * ndims; ++i) {
        initialLocation[i] = coord[i];
    }
    DMSwarmRestoreField(dm, DMSwarmPICField_coor, NULL, NULL, (void **)&coord) >> checkError;
    DMSwarmRestoreField(dm, ParticleInitialLocation, NULL, NULL, (void **)&initialLocation) >> checkError;
}

PetscErrorCode ablate::particles::Particles::ComputeParticleError(TS particleTS, Vec u, Vec errorVec) {
    PetscFunctionBeginUser;
    // get a pointer to this particle class
    ablate::particles::Particles* particles;
    TSGetApplicationContext(particleTS, (void **)&particles) >> checkError;

    // get the abs time for the particle evaluation, this is the ts relative time plus the time at the start of the particle ts solve
    PetscReal time;
    TSGetTime(particleTS, &time) >> checkError;
    time += particles->timeInitial;

    // Create a vector of the currentExactLocations
    DMSwarmVectorDefineField(particles->dm, ParticleInitialLocation) >> checkError;
    Vec exactLocationVec;
    DMGetGlobalVector(particles->dm, &exactLocationVec);
    PetscScalar *exactLocationArray;
    VecGetArrayWrite(exactLocationVec, &exactLocationArray) >> checkError;

    // exact the exact solution from the initial location
    PetscInt np;
    DMSwarmGetLocalSize(particles->dm, &np) >> checkError;
    const PetscInt dim = particles->ndims;

    // get the initial location array
    const PetscScalar *initialParticleLocationArray;
    DMSwarmGetField(particles->dm, ParticleInitialLocation, NULL, NULL, (void **)&initialParticleLocationArray) >> checkError;

    // for each local particle
    for (PetscInt p = 0; p < np; ++p) {
        PetscScalar x[3];
        PetscReal x0[3];
        PetscInt d;

        for (d = 0; d < dim; ++d) {
            x0[d] = PetscRealPart(initialParticleLocationArray[p * dim + d]);
        }

        particles->exactSolution->GetPetscFunction()(dim, time, x0, 1, x, particles->exactSolution->GetContext()) >> checkError;

        for (d = 0; d < dim; ++d) {
            exactLocationArray[p * dim + d] = x[d];
        }
    }
    VecRestoreArrayWrite(exactLocationVec, &exactLocationArray) >> checkError;

    // Get all points still in this mesh
    DM flowDM;
    VecGetDM(particles->flowFinal, &flowDM)  >> checkError;
    PetscSF cellSF = NULL;
    DMLocatePoints(flowDM, exactLocationVec, DM_POINTLOCATION_NONE, &cellSF) >> checkError;
    const PetscSFNode *cells;
    PetscSFGetGraph(cellSF, NULL, NULL, NULL, &cells) >> checkError;

    // compute the difference between exact and u
    VecWAXPY(errorVec, -1, exactLocationVec, u);

    // zero out the error if any particle moves outside of the domain
    for (PetscInt p = 0; p < np; ++p) {
        PetscInt d;
        if (cells[p].index == DMLOCATEPOINT_POINT_NOT_FOUND) {
            for (d = 0; d < dim; ++d) {
                VecSetValue(errorVec, p * dim + d, 0.0, INSERT_VALUES) >> checkError;
            }
        }
    }
    VecAssemblyBegin(errorVec) >> checkError;
    VecAssemblyEnd(errorVec) >> checkError;

    // restore all of the vecs/fields
    PetscSFDestroy(&cellSF) >> checkError;

    // cleanup
    DMSwarmRestoreField(particles->dm, ParticleInitialLocation, NULL, NULL, (void **)&initialParticleLocationArray) >> checkError;
    DMRestoreGlobalVector(particles->dm, &exactLocationVec) >> checkError;
    PetscFunctionReturn(0);
}
