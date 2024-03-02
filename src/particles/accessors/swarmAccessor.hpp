#ifndef ABLATELIBRARY_SWARMACCESSOR_HPP
#define ABLATELIBRARY_SWARMACCESSOR_HPP

#include <petsc.h>
#include <map>
#include "accessor.hpp"
#include "particles/field.hpp"
#include "utilities/petscUtilities.hpp"

namespace ablate::particles::accessors {
/**
 * class that will be passed to each processes to allow getting required data
 */
class SwarmAccessor : public Accessor<const PetscReal> {
   private:
    //! borrowed reference to
    const DM& swarmDm;

    //! a map of fields for easy field lookup
    const std::map<std::string, Field>& fieldsMap;

    //! The solution vector currently used in the ts
    Vec solutionVec;

    //! the array for the solution values
    const PetscScalar* solutionValues{};

   public:
    SwarmAccessor(bool cachePointData, const DM& swarmDm, const std::map<std::string, Field>& fieldsMap, Vec solutionVec)
        : Accessor(cachePointData), swarmDm(swarmDm), fieldsMap(fieldsMap), solutionVec(solutionVec) {
        // extract the array from the vector
        VecGetArrayRead(solutionVec, &solutionValues) >> utilities::PetscUtilities::checkError;
    }

    ~SwarmAccessor() override { VecRestoreArrayRead(solutionVec, &solutionValues) >> utilities::PetscUtilities::checkError; }

    /**
     * Returns the local size of the particles destination
     * @return
     */
    [[nodiscard]] inline PetscInt GetNumberParticles() const {
        PetscInt size;
        DMSwarmGetLocalSize(swarmDm, &size) >> utilities::PetscUtilities::checkError;
        return size;
    }

    /**
     * prevent copy of this class
     */
    SwarmAccessor(const SwarmAccessor&) = delete;

   protected:
    /**
     * Create point data from the solution field or swarmdm
     * @param fieldName
     * @return
     */
    ConstPointData CreateData(const std::string& fieldName) override {
        const auto& field = fieldsMap.at(fieldName);
        if (field.location == domain::FieldLocation::SOL) {
            return {solutionValues, field};
        } else {
            // get the field from the dm
            PetscScalar* values;
            DMSwarmGetField(swarmDm, field.name.c_str(), nullptr, nullptr, (void**)&values) >> utilities::PetscUtilities::checkError;

            // Register the cleanup
            RegisterCleanupFunction([=]() {
                const std::string name = field.name;
                DMSwarmRestoreField(swarmDm, name.c_str(), nullptr, nullptr, (void**)&values) >> utilities::PetscUtilities::checkError;
            });

            return {values, field};
        }
    }
};
}  // namespace ablate::particles::accessors
#endif  // ABLATELIBRARY_SWARMACCESSOR_HPP
