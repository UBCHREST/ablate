#ifndef ABLATELIBRARY_EDGECLUSTERINGMAPPER_HPP
#define ABLATELIBRARY_EDGECLUSTERINGMAPPER_HPP

#include "meshMapper.hpp"
#include "modifier.hpp"
namespace ablate::domain::modifiers {

/**
 * Performs clustering mapping using an algebraic relationship at the edges of the domain using Equation 9-42 from Hoffmann, Klaus A., and Steve T. Chiang. "Computational fluid dynamics volume I.
 * Forth Edition" Engineering education system (2000).
 *
 */
class EdgeClusteringMapper : public MeshMapper {
   private:
    //! The direction (0, 1, 2) to perform the mapping
    const int direction;
    //! The start of the domain in direction
    const double start;
    //! The size of the domain in direction
    const double size;
    //! The clustering factor
    const double beta;

   public:
    /**
     * Performs one point clustering
     * @param direction The direction (0, 1, 2) to perform the mapping
     * @param start The start of the domain in direction
     * @param end The end of the domain in direction
     * @param beta The clustering factor
     */
    explicit EdgeClusteringMapper(int direction, double start, double end, double beta);

    /**
     * Provide name of modifier for debug/output
     * @return
     */
    std::string ToString() const override;

   private:
    static PetscErrorCode MappingFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);
};
}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_ONEPOINTCLUSTERINGMAPPER_HPP
