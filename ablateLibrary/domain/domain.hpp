#ifndef ABLATELIBRARY_DOMAIN_H
#define ABLATELIBRARY_DOMAIN_H

#include <petsc.h>
#include <map>
#include <memory>
#include <string>
#include "fieldDescriptor.hpp"
#include "region.hpp"

namespace ablate::solver {
// forward declare the Solver
class Solver;
}  // namespace ablate::solver

namespace ablate::domain {
// forward declare the subDomain
class SubDomain;

class Domain : public std::enable_shared_from_this<Domain> {
   protected:
    Domain(std::string name);
    virtual ~Domain();

    std::string name;
    // The primary dm
    DM dm;

   private:
    // This domain can be partitions into multiple subdomains
    std::map<std::size_t, std::shared_ptr<SubDomain>> subDomains;

    // The aux vector DM
    DM auxDM;

    // The solution to the flow
    Vec solField;

    // The aux field to the flow
    Vec auxField;

    void CreateGlobalStructures();

    std::shared_ptr<SubDomain> GetSubDomain(std::shared_ptr<Region> name);

   public:
    std::string GetName() const { return name; }

    DM& GetDM() { return dm; }

    DM GetAuxDM() { return auxDM; }

    Vec GetSolutionVector() { return solField; }

    Vec GetAuxVector() { return auxField; }

    void RegisterField(const FieldDescriptor& fieldDescriptor, PetscObject field, DMLabel label);

    PetscInt GetDimensions() const;

    void InitializeSubDomains(std::vector<std::shared_ptr<solver::Solver>> solvers);
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_DOMAIN_H
