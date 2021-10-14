#include "subDomain.hpp"


ablate::domain::SubDomain::SubDomain(std::weak_ptr<Domain> domain, DMLabel label): domain(domain), label(label) {}


ablate::domain::Field ablate::domain::SubDomain::RegisterField(const ablate::domain::FieldDescriptor& fieldDescriptor, PetscObject field) {
    // Create a field with this information
    Field newField{.fieldName = fieldDescriptor.fieldName,
                   .components = fieldDescriptor.components,
                   .componentNames = fieldDescriptor.componentNames,
                   .fieldId = -1,
                   .fieldLocation = fieldDescriptor.fieldLocation};

    // Store the location in this subdomain
    newField.fieldId = fields.size();
    fields[newField.fieldName] = newField;

    if (auto domainPtr = domain.lock()) {
        domainPtr->RegisterField(fieldDescriptor, field, label);
    } else {
        throw std::runtime_error("Cannot RegisterField " + newField.fieldName + ". Domain is expired.");
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
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetAuxDM();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

Vec ablate::domain::SubDomain::GetSolutionVector() {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetSolutionVector();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

Vec ablate::domain::SubDomain::GetAuxVector() {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetAuxVector();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}

PetscInt ablate::domain::SubDomain::GetDimensions() const {
    if (auto domainPtr = domain.lock()) {
        return domainPtr->GetDimensions();
    } else {
        throw std::runtime_error("Cannot Get DM. Domain is expired.");
    }
}
