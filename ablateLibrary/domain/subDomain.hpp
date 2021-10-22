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

    // Keep a name for this subDomain for output/debug
    std::string name;

    // The label used to describe this subDomain
    DMLabel label;

    // Keep track of fields that live in this subDomain;
    std::map<std::string, Field> fields;

    // Each subDomain will operate over a ds
    PetscDS discreteSystem;

    // The auxDm and auxVec are for only this subDomain
    DM auxDM;
    Vec auxVec;

   public:
    SubDomain(std::weak_ptr<Domain> domain, std::shared_ptr<domain::Region>);
    ~SubDomain();

    Field RegisterField(const FieldDescriptor& fieldDescriptor, PetscObject field);

    /**
     * Create the auxDM, auxVec, and other structures on the subDomain
     */
    void CreateSubDomainStructures();

    // Returns the local field information
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

    // Get the discreteSystem describe system for this subDomain
    inline PetscDS GetDiscreteSystem(){
        return discreteSystem;
    }

    PetscInt GetDimensions() const;
    inline PetscInt GetNumberFields() const { return fields.size(); }

    inline MPI_Comm GetComm() { return PetscObjectComm((PetscObject)GetDM()); }
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_SUBDOMAIN_HPP
