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
    Domain(DM dm, std::string name, std::vector<std::shared_ptr<FieldDescriptor>>, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,
           const std::shared_ptr<parameters::Parameters>& options = {});
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

    //! The global solution vector defined over the entire dm. This is the canonical source of information
    Vec solGlobalField;

    void CreateStructures();

    // keep a list of functions that modify the dm
    std::vector<std::shared_ptr<modifiers::Modifier>> modifiers;

    //! the petsc options object to be applied to the main dm.
    PetscOptions petscOptions = nullptr;

   public:
    [[nodiscard]] std::string GetName() const { return name; }

    inline DM& GetDM() noexcept { return dm; }

    /**
     * Returns access to the global solution field
     * @return
     */
    inline Vec GetSolutionVector() { return solGlobalField; }

    /**
     * Register the field with the dm
     * @param fieldDescription
     */
    void RegisterField(const ablate::domain::FieldDescription& fieldDescription);

    [[nodiscard]] PetscInt GetDimensions() const noexcept;

    void InitializeSubDomains(const std::vector<std::shared_ptr<solver::Solver>>& solvers, const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& initializations,
                              const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& = {});

    /**
     * project the list of field function into the provided global vector
     * @param fieldFunctions
     * @param globVec
     */
    void ProjectFieldFunctions(const std::vector<std::shared_ptr<mathFunctions::FieldFunction>>& fieldFunctions, Vec globVec, PetscReal time = 0.0);

    std::shared_ptr<SubDomain> GetSubDomain(const std::shared_ptr<Region>& name);

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
    [[nodiscard]] inline const Field& GetField(int fieldId) const { return fields[fieldId]; }

    [[nodiscard]] inline const Field& GetField(const std::string& fieldName) const {
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
    [[nodiscard]] inline const std::vector<Field>& GetFields() const { return fields; }
};
}  // namespace ablate::domain
#endif  // ABLATELIBRARY_DOMAIN_H
