#ifndef ABLATELIBRARY_RhsDATA_HPP
#define ABLATELIBRARY_RhsDATA_HPP

#include <petsc.h>
#include <map>
#include "particles/field.hpp"
#include "utilities/petscError.hpp"
#include "particleData.hpp"

namespace ablate::particles {
/**
 * class that determines the rhs information
 */
class RhsData {
   private:
    //! a map of fields for easy field lookup
    const std::map<std::string_view, Field>& fieldsMap;

    //! The rhs vector currently used in the ts
    Vec rhsVec;

    //! the array for the rhs values
    PetscScalar* rhsValues;

   public:
    RhsData(const std::map<std::string_view, Field>& fieldsMap, Vec rhsVec) : fieldsMap(fieldsMap), rhsVec(rhsVec) {
        // extract the array from the vector
        VecGetArray(rhsVec, &rhsValues) >> checkError;
    }

    ~RhsData() { VecRestoreArray(rhsVec, &rhsValues) >> checkError; }

    /**
     * Get the field data for this field
     * @param fieldName
     * @return
     */
    ParticleData GetRHSData(std::string_view fieldName) {
        const auto& field = fieldsMap.at(fieldName);
        if (field.type == domain::FieldLocation::SOL) {
            return ParticleData(field, rhsValues);
        } else {
            throw std::invalid_argument("The field " + std::string(fieldName) + " is not a solution variable");
        }
    }

    /**
     * prevent copy of this class
     */
    RhsData(const RhsData&) = delete;
};
}  // namespace ablate::particles::processes
#endif  // ABLATELIBRARY_SWARMDATA_HPP
