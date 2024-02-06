#ifndef ABLATELIBRARY_EULERIANSOURCEACCESSOR_HPP
#define ABLATELIBRARY_EULERIANSOURCEACCESSOR_HPP

#include <petsc.h>
#include <map>
#include "accessor.hpp"
#include "domain/subDomain.hpp"
#include "particles/field.hpp"
#include "swarmAccessor.hpp"
#include "utilities/petscUtilities.hpp"

namespace ablate::particles::accessors {
/**
 * Allows pushing source terms to the particle source array based upon variable/component name.
 * The coupled solver is used to push back to the flow field ts
 */
class EulerianSourceAccessor : public Accessor<PetscReal> {
   public:
    // A string to hold the coupled source terms name
    inline static const char CoupledSourceTermPostfix[] = "_CoupledSourceTerm";

   private:
    //! borrowed reference to
    const DM& swarmDm;

    //! a map of fields for easy field lookup
    const std::map<std::string, Field>& fieldsMap;

   public:
    EulerianSourceAccessor(bool cachePointData, const DM& swarmDm, const std::map<std::string, Field>& fieldsMap) : Accessor(cachePointData), swarmDm(swarmDm), fieldsMap(fieldsMap) {}

    /**
     * Create point data from the source field in the DM
     * @param fieldName
     * @return
     */
    PointData CreateData(const std::string& fieldName) override {
        const auto& field = fieldsMap.at(fieldName + CoupledSourceTermPostfix);
        if (field.location == domain::FieldLocation::SOL) {
            throw std::invalid_argument("Eulerian Source Fields should not be SOL fields");
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

    /**
     * prevent copy of this class
     */
    EulerianSourceAccessor(const EulerianSourceAccessor&) = delete;
};
}  // namespace ablate::particles::accessors
#endif  // ABLATELIBRARY_EULERIANSOURCEACCESSOR_HPP
