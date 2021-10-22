#include "subDomain.hpp"
#include <utilities/petscError.hpp>

ablate::domain::SubDomain::SubDomain(std::weak_ptr<Domain> domain, std::shared_ptr<domain::Region> region) : domain(domain), region(region), name(""), label(nullptr), auxDM(nullptr), auxVec(nullptr) {
    if (region) {
        if (auto domainPtr = domain.lock()) {
            DMGetLabel(domainPtr->GetDM(), region->GetName().c_str(), &label) >> checkError;
        } else {
            throw std::runtime_error("Cannot Locate Field in DM. DM is NULL");
        }

        name = region->GetName();
    }
}

ablate::domain::SubDomain::~SubDomain() {
    if (auxDM) {
        DMDestroy(&auxDM) >> checkError;
    }

    if (auxVec) {
        VecDestroy(&auxVec) >> checkError;
    }
}


ablate::domain::Field ablate::domain::SubDomain::RegisterField(const ablate::domain::FieldDescriptor& fieldDescriptor, PetscObject field) {
    // Create a field with this information
    Field newField{.name = fieldDescriptor.name, .numberComponents = (PetscInt)fieldDescriptor.components.size(), .components = fieldDescriptor.components, .id = -1, .type = fieldDescriptor.type};

    // Store the location in this subdomain
    auto type = fieldDescriptor.type;
    newField.id = std::count_if(fields.begin(), fields.end(), [type](auto pair) { return pair.second.type == type; });
    fields[newField.name] = newField;

    if (auto domainPtr = domain.lock()) {
        // add solution fields/aux fields
        switch (fieldDescriptor.type) {
            case FieldType::SOL: {
                domainPtr->RegisterSolutionField(fieldDescriptor, field, label);
                break;
            }
            case FieldType::AUX: {
                // check to see if need to create an aux dm
                if (auxDM == nullptr) {
                    /* MUST call DMGetCoordinateDM() in order to get p4est setup if present */
                    DM coordDM;
                    DMGetCoordinateDM(domainPtr->GetDM(), &coordDM) >> checkError;
                    DMClone(domainPtr->GetDM(), &auxDM) >> checkError;

                    // this is a hard coded "dmAux" that petsc looks for
                    DMSetCoordinateDM(auxDM, coordDM) >> checkError;
                }
                DMAddField(auxDM, label, (PetscObject)field) >> checkError;
            }
        }

    } else {
        throw std::runtime_error("Cannot RegisterField " + newField.name + ". Domain is expired.");
    }




    return newField;
}

DM& ablate::domain::SubDomain::GetDM() {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetDM();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

DM ablate::domain::SubDomain::GetAuxDM() {
    return auxDM;
}

Vec ablate::domain::SubDomain::GetSolutionVector() {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetSolutionVector();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

Vec ablate::domain::SubDomain::GetAuxVector() {
    return auxVec;
}

PetscInt ablate::domain::SubDomain::GetDimensions() const {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetDimensions();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

void ablate::domain::SubDomain::CreateSubDomainStructures(){
    if (auto domainPtr = domain.lock()) {
        if (auxDM) {
            DMCreateDS(auxDM) >> checkError;
            DMCreateLocalVector(auxDM, &(auxVec)) >> checkError;

            // attach this field as aux vector to the dm
            DMSetAuxiliaryVec(domainPtr->GetDM(), label, label? region->GetValues()[0] : 0, auxVec) >> checkError;
            auto vecName = "aux" +( region?  "_" +  region->GetName() : "");
            PetscObjectSetName((PetscObject)auxVec, vecName.c_str()) >> checkError;
        }
    } else {
        throw std::runtime_error("Cannot CreateSubDomainStructures. Domain is expired.");
    }

}