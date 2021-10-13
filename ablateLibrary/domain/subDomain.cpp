#include "subDomain.hpp"

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
