#ifndef ABLATELIBRARY_EULERIANDATA_HPP
#define ABLATELIBRARY_EULERIANDATA_HPP

#include <petsc.h>
#include <map>
#include "accessor.hpp"
#include "domain/subDomain.hpp"
#include "particles/field.hpp"
#include "swarmAccessor.hpp"
#include "utilities/petscError.hpp"

namespace ablate::particles::accessors {
/**
 * Interpolates cell/eulerian data to the particle locations
 */
class EulerianAccessor : public Accessor<const PetscReal> {
   private:
    const std::shared_ptr<ablate::domain::SubDomain> subDomain;

    //! current time in the solver
    const PetscReal currentTime;

    //! Store a list of current coordinates
    std::vector<PetscReal> coordinates;

    //! the number of particles in this domain
    const PetscInt np;

   public:
    EulerianAccessor(bool cachePointData, std::shared_ptr<ablate::domain::SubDomain> subDomain, SwarmAccessor&, PetscReal currentTime);

    /**
     * Create point data from the rhs field
     * @param fieldName
     * @return
     */
    ConstPointData CreateData(const std::string& fieldName) override;

    /**
     * prevent copy of this class
     */
    EulerianAccessor(const EulerianAccessor&) = delete;

    /**
     * Get the number of dimensions
     */
    inline PetscInt GetDimensions() const { return subDomain->GetDimensions(); }
};
}  // namespace ablate::particles::accessors
#endif  // ABLATELIBRARY_SWARMDATA_HPP
