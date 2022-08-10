#ifndef ABLATELIBRARY_EULERIANDATA_HPP
#define ABLATELIBRARY_EULERIANDATA_HPP

#include <petsc.h>
#include <map>
#include "particles/field.hpp"
#include "utilities/petscError.hpp"
#include "particleData.hpp"

namespace ablate::particles {
/**
 * Interpolates cell/eulerian data to the particle locations
 */
class EulerianData {
   private:
    //! borrowed reference to
    const DM& swarmDm;

    //! a map of fields for easy field lookup
    const std::map<std::string_view, Field>& fieldsMap;

    //! The solution vector currently used in the ts
    Vec solutionVec;

    //! the array for the solution values
    PetscScalar* solutionValues;

   public:
    EulerianData(const DM& swarmDm, const std::map<std::string_view, Field>& fieldsMap, Vec solutionVec) : swarmDm(swarmDm), fieldsMap(fieldsMap), solutionVec(solutionVec) {
        // extract the array from the vector
        VecGetArray(solutionVec, &solutionValues) >> checkError;
    }

    ~EulerianData() { VecRestoreArray(solutionVec, &solutionValues) >> checkError; }

    /**
     * Get the field data for this field
     * @param fieldName
     * @return
     */
    ParticleData GetFieldData(std::string_view fieldName) {
        const auto& field = fieldsMap.at(fieldName);
        if (field.type == domain::FieldLocation::SOL) {
            return ParticleData(field, solutionValues);
        } else {
            return ParticleData(field, swarmDm);
        }
    }


    /**
     * Get the field data for this field
     * @param fieldName
     * @return
     */
    const ParticleData GetFieldData(std::string_view fieldName) const {
        const auto& field = fieldsMap.at(fieldName);
        if (field.type == domain::FieldLocation::SOL) {
            return ParticleData(field, solutionValues);
        } else {
            return ParticleData(field, swarmDm);
        }
    }

    /**
     * prevent copy of this class
     */
    SwarmData(const SwarmData&) = delete;
};
}  // namespace ablate::particles::processes
#endif  // ABLATELIBRARY_SWARMDATA_HPP
