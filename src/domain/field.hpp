#ifndef ABLATELIBRARY_FIELD_HPP
#define ABLATELIBRARY_FIELD_HPP

#include <petsc.h>
#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace ablate::domain {

enum class FieldLocation { SOL, AUX };
enum class FieldType { FVM, FEM };

struct FieldDescription;

struct Field {
    const std::string name;
    const PetscInt numberComponents;
    const std::vector<std::string> components;

    // The global id in the dm for this field
    const PetscInt id;

    // The specific id in this ds/subdomain
    const PetscInt subId = PETSC_DEFAULT;

    // The specific offset for this field in this ds/subdomain
    const PetscInt offset = PETSC_DEFAULT;

    const enum FieldLocation location = FieldLocation::SOL;

    // Keep track of the field type
    const enum FieldType type;

    // store any optional tags, there are strings that can be used to describe the field
    const std::set<std::string> tags;

    static Field FromFieldDescription(const FieldDescription& fieldDescription, PetscInt id, PetscInt subId = PETSC_DEFAULT, PetscInt offset = PETSC_DEFAULT);

    [[nodiscard]] Field CreateSubField(PetscInt subId, PetscInt offset) const;

    // helper function to check if the field contains a certain tag
    inline bool Tagged(std::string_view tag) const {
        return std::any_of(tags.begin(), tags.end(), [tag](const auto& tagItem) { return tagItem == tag; });
    }

    /**
     * return the component index in the current field
     * @param search
     * @return
     */
    inline std::size_t ComponentIndex(std::string_view search) const {
        auto it = std::find_if(components.begin(), components.end(), [search](const auto& component) { return component == search; });

        // If element was found
        if (it != components.end()) {
            return std::distance(components.begin(), it);
        } else {
            throw std::invalid_argument(std::string("Cannot locate component ") + std::string(search) + " in field " + name);
        }
    }

    /**
     * return the component offset (field offset + index) in the current field/component
     * @param search
     * @return
     */
    inline std::size_t ComponentOffset(std::string_view search) const { return ComponentIndex(search) + offset; }
};

std::istream& operator>>(std::istream& is, FieldLocation& v);
std::istream& operator>>(std::istream& is, FieldType& v);

}  // namespace ablate::domain
#endif  // ABLATELIBRARY_FIELD_HPP
