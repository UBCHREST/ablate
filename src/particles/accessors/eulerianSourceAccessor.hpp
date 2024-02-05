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
    inline static const char CoupledSourceTerm[] = "CoupledSourceTerm";

   private:
    const std::shared_ptr<ablate::domain::SubDomain> subDomain;

    //! borrowed reference to
    const DM& swarmDm;

   public:
    EulerianSourceAccessor(bool cachePointData, std::shared_ptr<ablate::domain::SubDomain> subDomain, const DM& swarmDm);

    /**
     * Create point data from the rhs field
     * @param fieldName
     * @return
     */
    PointData CreateData(const std::string& fieldName) override;

    /**
     * prevent copy of this class
     */
    EulerianSourceAccessor(const EulerianSourceAccessor&) = delete;
};
}  // namespace ablate::particles::accessors
#endif  // ABLATELIBRARY_EULERIANSOURCEACCESSOR_HPP
