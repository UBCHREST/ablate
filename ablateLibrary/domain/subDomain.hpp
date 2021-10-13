#ifndef ABLATELIBRARY_SUBDOMAIN_HPP
#define ABLATELIBRARY_SUBDOMAIN_HPP

#include <memory>
#include "domain.hpp"
#include "fieldDescriptor.hpp"
#include <map>
#include <string>

namespace ablate::domain {

class SubDomain {
   private:
    // const pointer to the parent domain/dm
    const std::weak_ptr<Domain> domain;

    // The label used to describe this subDomain
    DMLabel label;

    // Keep track of fields that live in this subDomain;
    std::map<std::string, Field> fields;

   public:
    Field RegisterField(const FieldDescriptor& fieldDescriptor, PetscObject field);
};

}
#endif  // ABLATELIBRARY_SUBDOMAIN_HPP
