#include "domain.hpp"
#include <utilities/mpiError.hpp>
#include "solver/solver.hpp"
#include "subDomain.hpp"
#include "utilities/petscError.hpp"

ablate::domain::Domain::Domain(std::string name) : name(name), solField(nullptr) {}

ablate::domain::Domain::~Domain() {
    // clean up the petsc objects
    if (solField) {
        VecDestroy(&solField) >> checkError;
    }
}

void ablate::domain::Domain::RegisterSolutionField(const ablate::domain::FieldDescriptor& fieldDescriptor, PetscObject field, DMLabel label) {
    // add solution fields/aux fields
    switch (fieldDescriptor.type) {
        case FieldType::SOL: {
            // Called the shared method to register
            DMAddField(dm, label, (PetscObject)field) >> checkError;

            // Copy to a field Field
            Field newField{.name = fieldDescriptor.name,
                           .numberComponents = (PetscInt)fieldDescriptor.components.size(),
                           .components = fieldDescriptor.components,
                           .id = (PetscInt)solutionFields.size(),
                           .type = fieldDescriptor.type};

            break;
        }
        default:
            throw std::runtime_error("Can only register SOL fields in Domain::RegisterSolutionField");
    }
}

PetscInt ablate::domain::Domain::GetDimensions() const {
    PetscInt dim;
    DMGetDimension(dm, &dim) >> checkError;
    return dim;
}

void ablate::domain::Domain::CreateStructures() {
    // Setup the solve with the ts
    DMPlexCreateClosureIndex(dm, NULL) >> checkError;
    DMCreateGlobalVector(dm, &(solField)) >> checkError;
    PetscObjectSetName((PetscObject)solField, "solution") >> checkError;
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
    CreateStructures();
    for(auto& subDomain: subDomains){
        subDomain.second->CreateSubDomainStructures();
    }


    // Initialize each solver
    for (auto& solver : solvers) {
        solver->Initialize();
    }
}
