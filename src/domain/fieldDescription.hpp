#ifndef ABLATELIBRARY_FIELDDESCRIPTION_HPP
#define ABLATELIBRARY_FIELDDESCRIPTION_HPP

#include <petsc.h>
#include <algorithm>
#include <memory>
#include <parameters/parameters.hpp>
#include <string>
#include <vector>
#include "domain/field.hpp"
#include "domain/region.hpp"
#include "fieldDescriptor.hpp"

namespace ablate::domain {

/**
 * Describes the necessary information to produce a field in the domain/dm
 */
struct FieldDescription : public FieldDescriptor, public std::enable_shared_from_this<FieldDescription> {
    virtual ~FieldDescription();

    // Helper variable, replaces any components with this value with one for each dimension
    inline const static std::string DIMENSION = "_DIMENSION_";
    inline const static std::vector<std::string> ONECOMPONENT = {"_"};

    // The name of the field
    const std::string name;

    // The prefix for field options
    const std::string prefix;

    // The components held in this field
    std::vector<std::string> components = {"_"};

    // If the field is solution or aux
    const enum FieldLocation location = FieldLocation::SOL;

    // The type of field (FEM/FVM)
    const enum FieldType type = FieldType::FEM;

    // The region for the field (nullptr is everywhere)
    const std::shared_ptr<domain::Region> region;

    // store any optional tags, there are strings that can be used to describe the field
    const std::vector<std::string> tags;

    FieldDescription(std::string name, std::string prefix, std::vector<std::string> components, FieldLocation location, FieldType type, std::shared_ptr<domain::Region> = {},
                     std::shared_ptr<parameters::Parameters> = {}, std::vector<std::string> tags = {});

    /**
     * Public function that will cause the components to expand or decompress based upon the number of dims
     */
    void DecompressComponents(PetscInt dim);

    /** Allow a single FieldDescription to report it self allowing FieldDescription to be used as FieldDescriptor**/
    std::vector<std::shared_ptr<FieldDescription>> GetFields() override;

    /**
     * Create the petsc field used by the dm
     * @param dm
     * @return
     */
    virtual PetscObject CreatePetscField(DM dm) const;

   private:
    // Petsc options specific for this field
    PetscOptions options = nullptr;
};

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELDDESCRIPTION_HPP
