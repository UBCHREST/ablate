#ifndef ABLATELIBRARY_DOMAIN_H
#define ABLATELIBRARY_DOMAIN_H

#include <petsc.h>
#include <map>
#include <string>

namespace ablate::mesh {
class Mesh {
   protected:
    std::string name;
    DM dm;

    Mesh(std::string name): name(name){};
    virtual ~Mesh() = default;

   public:
    std::string GetName() const { return name; }

    DM& GetDomain() { return dm; }
};
}  // namespace ablate::mesh
#endif  // ABLATELIBRARY_DOMAIN_H
