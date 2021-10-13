#ifndef ABLATELIBRARY_DOMAIN_H
#define ABLATELIBRARY_DOMAIN_H

#include <petsc.h>
#include <map>
#include <memory>
#include <string>
#include "fieldDescriptor.hpp"

namespace ablate::domain {
// forward declare the subDomain
class SubDomain;

class Domain {
   protected:
    Domain(std::string name);
    virtual ~Domain();

    std::string name;
    // The primary dm
    DM dm;

   private:
    // This domain can be partitions into multiple subdomains
    std::map<std::string, std::shared_ptr<SubDomain>> subDomains;

    // The aux vector DM
    DM auxDM;

    // The solution to the flow
    Vec solField;

    // The aux field to the flow
    Vec auxField;

   public:
    std::string GetName() const { return name; }

    DM& GetDomain() { return dm; }

    int GetDimensions() const;

    void RegisterField(const FieldDescriptor& fieldDescriptor, PetscObject field, DMLabel label);

    // TODO: Remove call
    void FinalizeRegisterFields();


};
}  // namespace ablate::mesh
#endif  // ABLATELIBRARY_DOMAIN_H
