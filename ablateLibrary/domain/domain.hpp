#ifndef ABLATELIBRARY_DOMAIN_H
#define ABLATELIBRARY_DOMAIN_H

#include <petsc.h>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
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
    // Keep track of all solution fields
    std::map<std::string, Field> solutionFields;

    // This domain can be partitions into multiple subdomains
    std::map<std::size_t, std::shared_ptr<SubDomain>> subDomains;

    // The solution to the flow
    Vec solField;

    void CreateStructures();

    std::shared_ptr<SubDomain> GetSubDomain(std::shared_ptr<Region> name);

   public:
    std::string GetName() const { return name; }

    DM &GetDM() { return dm; }

    Vec GetSolutionVector() { return solField; }

    void RegisterSolutionField(const FieldDescriptor &fieldDescriptor, PetscObject field, DMLabel label);

    PetscInt GetDimensions() const;

    void InitializeSubDomains(std::vector<std::shared_ptr<solver::Solver>> solvers);

    inline const Field &GetSolutionField(const std::string &fieldName) const {
        if (solutionFields.count(fieldName)) {
            return solutionFields.at(fieldName);
        } else {
            throw std::invalid_argument("Cannot locate field " + fieldName + " in subDomain " + name);
        }
    }
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_DOMAIN_H
