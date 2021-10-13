#ifndef ABLATELIBRARY_DOMAIN_H
#define ABLATELIBRARY_DOMAIN_H

#include <petsc.h>
#include <map>
#include <string>

namespace ablate::domain {
class Domain {
   protected:
    std::string name;
    DM dm;

    Domain(std::string name) : name(name){};
    virtual ~Domain() = default;

   public:
    std::string GetName() const { return name; }

    DM& GetDomain() { return dm; }

    int GetDimensions() const;
};
}  // namespace ablate::mesh
#endif  // ABLATELIBRARY_DOMAIN_H
