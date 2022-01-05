#ifndef ABLATELIBRARY_DOMAIN_H
#define ABLATELIBRARY_DOMAIN_H
#include <petsc.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "domain/fieldDescriptor.hpp"
#include "domain/modifiers/modifier.hpp"
#include "fieldDescription.hpp"
#include "io/serializable.hpp"
#include "mathFunctions/fieldFunction.hpp"
#include "region.hpp"

namespace ablate::solver {
// forward declare the Solver
class Solver;
}  // namespace ablate::solver

namespace ablate::domain {
// forward declare the subDomain
class SubDomain;

class Domain {
   protected:
    Domain(DM dm, std::string name, std::vector<std::shared_ptr<FieldDescriptor>>, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers);
    virtual ~Domain();

    // The primary dm
    DM dm;

   private:
    // the name of the dm
    std::string name;

    // Hold a copy of the comm for this DM
    const MPI_Comm comm;

    // List of classes that are used to describe fields
    const std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors;

    // Keep track of all solution fields
    std::vector<Field> fields;

    // This domain can be partitions into multiple subdomains
    std::vector<std::shared_ptr<SubDomain>> subDomains;

    // The solution to the flow
    Vec solField;

    void CreateStructures();

    // keep a list of functions that modify the dm
    std::vector<std::shared_ptr<modifiers::Modifier>> modifiers;

   public:
    std::string GetName() const { return name; }

    inline DM& GetDM() noexcept { return dm; }

    Vec GetSolutionVector() { return solField; }

    void RegisterField(const ablate::domain::FieldDescription& fieldDescription);

    PetscInt GetDimensions() const;

    void InitializeSubDomains(std::vector<std::shared_ptr<solver::Solver>> solvers, const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initializations,
                              const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& = {});

    std::shared_ptr<SubDomain> GetSubDomain(std::shared_ptr<Region> name);

    /**
     * Provide a list of serialize subDomains
     */
    std::vector<std::weak_ptr<io::Serializable>> GetSerializableSubDomains();

    /**
     * Get the petscField object from the dm or auxDm for this region
     * @param fieldName
     * @return
     */
    PetscObject GetPetscFieldObject(const Field& field);

    /**
     *  returns the field  by global id
     * @param fieldId
     * @return
     */
    inline const Field& GetField(int fieldId) const { return fields[fieldId]; }

    inline const Field& GetField(const std::string& fieldName) const {
        auto field = std::find_if(fields.begin(), fields.end(), [&fieldName](auto field) { return field.name == fieldName; });
        if (field != fields.end()) {
            return *field;
        } else {
            throw std::invalid_argument("Cannot locate field with name " + fieldName + " in domain " + name);
        }
    }

    /**
     *  returns all of the fields
     * @param fieldId
     * @return
     */
    inline const std::vector<Field>& GetFields() const { return fields; }
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_DOMAIN_H
