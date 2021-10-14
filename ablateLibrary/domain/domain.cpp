#include "domain.hpp"
#include <utilities/mpiError.hpp>
#include "utilities/petscError.hpp"
#include "subDomain.hpp"

ablate::domain::Domain::Domain(std::string name) : name(name), auxDM(nullptr), solField(nullptr), auxField(nullptr){}

ablate::domain::Domain::~Domain(){
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
    switch(fieldDescriptor.fieldLocation){
        case FieldLocation::SOL:{
            // Called the shared method to register
            DMAddField(dm, label, (PetscObject)field) >> checkError;
            break;
        }
        case FieldLocation::AUX:{
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

void ablate::domain::Domain::CompleteSetup(TS ts) {
    DMCreateDS(dm) >> checkError;

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

std::shared_ptr<ablate::domain::SubDomain> ablate::domain::Domain::GetSubDomain(const std::string& subDomainName){
    if(subDomains.count(subDomainName) == 0){
        subDomains[subDomainName] = std::make_shared<ablate::domain::SubDomain>(shared_from_this(), nullptr);
    }
    return subDomains[subDomainName];
}
std::shared_ptr<ablate::domain::SubDomain> ablate::domain::Domain::GetSubDomain() {
    if(subDomains.count("") == 0){
        subDomains[""] = std::make_shared<ablate::domain::SubDomain>(shared_from_this(), nullptr);
    }
    return subDomains[""];
}
