#ifndef ABLATELIBRARY_DOMAIN_H
#define ABLATELIBRARY_DOMAIN_H

#include <petsc.h>
#include <string>
#include <map>

namespace ablate {
namespace mesh {
class Mesh {
   protected:
    std::string name;
    DM dm;
    MPI_Comm  comm;

    Mesh(MPI_Comm comm, std::string name, std::map<std::string, std::string> arguments);
    virtual ~Mesh();

   public:
    std::string GetName() const{
        return name;
    }

    DM GetDomain(){
        return dm;
    }
};
}
}
#endif  // ABLATELIBRARY_DOMAIN_H
