#ifndef ABLATELIBRARY_RHSACCESSOR_HPP
#define ABLATELIBRARY_RHSACCESSOR_HPP

#include <petsc.h>
#include <map>
#include "accessor.hpp"
#include "particles/field.hpp"
#include "utilities/petscError.hpp"

namespace ablate::particles::accessors {
/**
 * class that determines the rhs information
 */
class RhsAccessor : public Accessor<PetscReal> {
   private:
    //! a map of fields for easy field lookup
    const std::map<std::string, Field>& fieldsMap;

    //! The rhs vector currently used in the ts
    Vec rhsVec;

    //! the array for the rhs values
    PetscScalar* rhsValues;

   public:
    RhsAccessor(bool cachePointData, const std::map<std::string, Field>& fieldsMap, Vec rhsVec) : Accessor(cachePointData), fieldsMap(fieldsMap), rhsVec(rhsVec) {
        // extract the array from the vector
        VecGetArray(rhsVec, &rhsValues) >> checkError;
    }

    /**
     * clean up the rhs values
     */
    ~RhsAccessor() override { VecRestoreArray(rhsVec, &rhsValues) >> checkError; }

    /**
     * Create point data from the rhs field
     * @param fieldName
     * @return
     */
    PointData CreateData(const std::string& fieldName) override {
        const auto& field = fieldsMap.at(fieldName);
        if (field.type == domain::FieldLocation::SOL) {
            return PointData(rhsValues, field);
        } else {
            throw std::invalid_argument("The field " + std::string(fieldName) + " is not a solution variable");
        }
    }

    /**
     * prevent copy of this class
     */
    RhsAccessor(const RhsAccessor&) = delete;
};
}  // namespace ablate::particles::accessors
#endif  // ABLATELIBRARY_SWARMDATA_HPP
