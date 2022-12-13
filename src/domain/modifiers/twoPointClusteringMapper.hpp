#ifndef ABLATELIBRARY_TWOPOINTCLUSTERINGMAPPER_HPP
#define ABLATELIBRARY_TWOPOINTCLUSTERINGMAPPER_HPP

#include "meshMapper.hpp"
#include "modifier.hpp"
namespace ablate::domain::modifiers {

/**
 * Performs clustering mapping using an algebraic relationship around two point using equations derived from Hoffmann, Klaus A., and Steve T. Chiang. "Computational fluid dynamics volume I. Forth
 * Edition" Engineering education system (2000).
 */
class TwoPointClusteringMapper : public MeshMapper {
   private:
    //! The direction (0, 1, 2) to perform the mapping
    const int direction;
    //! The start of the domain in direction
    const double start;
    //! The size of the domain in direction
    const double size;
    //! The clustering factor
    const double beta;
    //! The location to cluster center
    const double location;
    //! The offset from the location center to perform the clustering
    const double offset;

   public:
    /**
     * Performs one point clustering
     * @param direction The direction (0, 1, 2) to perform the mapping
     * @param start The start of the domain in direction
     * @param end The end of the domain in direction
     * @param beta The clustering factor
     * @param location The location to cluster center
     * @param offset The location to perform the clustering in direction
     */
    explicit TwoPointClusteringMapper(int direction, double start, double end, double beta, double location, double offset);

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
