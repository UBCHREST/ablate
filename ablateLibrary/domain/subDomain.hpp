#ifndef ABLATELIBRARY_SUBDOMAIN_HPP
#define ABLATELIBRARY_SUBDOMAIN_HPP

#include <petsc.h>
#include <map>
#include <memory>
#include <string>
#include "domain.hpp"
#include "fieldDescriptor.hpp"

namespace ablate::domain {

class SubDomain {
   private:
    // const pointer to the parent domain/dm
    const std::weak_ptr<Domain> domain;

    // store the region used to define this subDomain
    const std::shared_ptr<domain::Region> region;

    // The label used to describe this subDomain
    DMLabel label;

    // Keep track of fields that live in this subDomain;
    std::map<std::string, Field> fields;

    // Keep a copy of this subDomain name.
    std::string name;

   public:
    SubDomain(std::weak_ptr<Domain> domain, std::shared_ptr<domain::Region>);

    Field RegisterField(const FieldDescriptor& fieldDescriptor, PetscObject field);

    inline const Field& GetField(const std::string& fieldName) const {
        if (fields.count(fieldName)) {
            return fields.at(fieldName);
        } else {
            throw std::invalid_argument("Cannot locate field " + fieldName + " in subDomain " + name);
        }
    }

    //[[deprecated("Should remove need for direct dm access")]]
    DM& GetDM();

    //[[deprecated("Should remove need for direct dm access")]]
    DM GetAuxDM();

    Vec GetSolutionVector();
    Vec GetAuxVector();

    PetscInt GetDimensions() const;
    inline PetscInt GetNumberFields() const { return fields.size(); }

    inline MPI_Comm GetComm() { return PetscObjectComm((PetscObject)GetDM()); }
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_SUBDOMAIN_HPP
