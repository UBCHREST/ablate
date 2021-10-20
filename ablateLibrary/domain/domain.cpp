#include "domain.hpp"
#include <utilities/mpiError.hpp>
#include "solver/solver.hpp"
#include "subDomain.hpp"
#include "utilities/petscError.hpp"

ablate::domain::Domain::Domain(std::string name) : name(name), auxDM(nullptr), solField(nullptr), auxField(nullptr) {}

ablate::domain::Domain::~Domain() {
    if (auxDM) {
        DMDestroy(&auxDM) >> checkError;
    }
    // clean up the petsc objects
    if (solField) {
        VecDestroy(&solField) >> checkError;
    }
    if (auxField) {
        VecDestroy(&auxField) >> checkError;
    }
}

void ablate::domain::Domain::RegisterField(const ablate::domain::FieldDescriptor& fieldDescriptor, PetscObject field, DMLabel label) {
    // add solution fields/aux fields
    switch (fieldDescriptor.type) {
        case FieldType::SOL: {
            // Called the shared method to register
            DMAddField(dm, label, (PetscObject)field) >> checkError;
            break;
        }
        case FieldType::AUX: {
            // check to see if need to create an aux dm
            if (auxDM == nullptr) {
                /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
                DM coordDM;
                DMGetCoordinateDM(dm, &coordDM) >> checkError;
                DMClone(dm, &auxDM) >> checkError;

                // this is a hard coded "dmAux" that petsc looks for
                PetscObjectCompose((PetscObject)dm, "dmAux", (PetscObject)auxDM) >> checkError;
                DMSetCoordinateDM(auxDM, coordDM) >> checkError;
            }
            DMAddField(auxDM, label, (PetscObject)field) >> checkError;
        }
    }
}

PetscInt ablate::domain::Domain::GetDimensions() const {
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;
    return dim;
}

void ablate::domain::Domain::CreateGlobalStructures() {
    // Setup the solve with the ts
    DMPlexCreateClosureIndex(dm, NULL) >> checkError;
    DMCreateGlobalVector(dm, &(solField)) >> checkError;
    PetscObjectSetName((PetscObject)solField, "flowField") >> checkError;

    if (auxDM) {
        DMCreateDS(auxDM) >> checkError;
        DMCreateLocalVector(auxDM, &(auxField)) >> checkError;

        // attach this field as aux vector to the dm
        DMSetAuxiliaryVec(dm, NULL, 0, auxField) >> checkError;
        PetscObjectSetName((PetscObject)auxField, "auxField") >> checkError;
    }
}

std::shared_ptr<ablate::domain::SubDomain> ablate::domain::Domain::GetSubDomain(std::shared_ptr<domain::Region> region) {
    std::size_t regionHash = region ? region->GetId() : 0;
    if (subDomains.count(regionHash) == 0) {
        subDomains[regionHash] = std::make_shared<ablate::domain::SubDomain>(shared_from_this(), nullptr);
    }
    return subDomains[regionHash];
}

void ablate::domain::Domain::InitializeSubDomains(std::vector<std::shared_ptr<solver::Solver>> solvers) {
    // determine the number of fields
    for (auto& solver : solvers) {
        solver->Register(GetSubDomain(solver->GetRegion()));
    }

    // Set up the global DS
    DMCreateDS(dm) >> checkError;

    // Setup each of the fields
    for (auto& solver : solvers) {
        solver->Setup();
    }

    // Create the global structures
    CreateGlobalStructures();

    // Initialize each solver
    for (auto& solver : solvers) {
        solver->Initialize();
    }
}
