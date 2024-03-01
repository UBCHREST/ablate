#ifndef ABLATELIBRARY_AXISDESCRIPTION_HPP
#define ABLATELIBRARY_AXISDESCRIPTION_HPP

#include <petsc.h>
#include <domain/region.hpp>
#include <set>

namespace ablate::domain::descriptions {
/**
 * A simple interface that describes points a long a line on axis
 */
class AxisDescription {
   public:
    virtual ~AxisDescription() = default;
    /**
     * Total number of nodes/vertices in the entire mesh
     * @return
     */
    [[nodiscard]] virtual const PetscInt& GetNumberVertices() const = 0;

    /**
     * Builds the node coordinate for each vertex
     * @return
     */
    virtual void SetCoordinate(PetscInt node, PetscReal coordinate[3]) const = 0;
};
}  // namespace ablate::domain::descriptions

#endif  // ABLATELIBRARY_AXISDESCRIPTION_HPP
