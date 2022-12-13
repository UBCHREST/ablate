#ifndef ABLATELIBRARY_ONEPOINTCLUSTERINGMAPPER_HPP
#define ABLATELIBRARY_ONEPOINTCLUSTERINGMAPPER_HPP

#include "meshMapper.hpp"
#include "modifier.hpp"
namespace ablate::domain::modifiers {

/**
 * Performs clustering mapping using an algebraic relationship around one point using Equation 9-50 from
 *
 * Hoffmann, Klaus A., and Steve T. Chiang. "Computational fluid dynamics volume I. Forth Edition" Engineering education system (2000).
 *
 * $$x'=D{ 1+\frac{sinh[\beta(x-A))]}{sinh(\beta A)}}$$
 *
 * Where:
 *
 * $$ A=\frac{1}{2 \beta}ln \left [  \frac{1+(e^\beta - 1)(D/H)}{1+(e^{-\beta} - 1)(D/H)} \right ] $$
 * $$ D = cluster location$$
 * $$ \beta = cluster factor $$
 *
 */
class OnePointClusteringMapper : public MeshMapper {
   private:
    //! The direction (0, 1, 2) to perform the mapping
    const int direction;
    //! The start of the domain in direction
    const double start;
    //! The size of the domain in direction
    const double size;
    //! The clustering factor
    const double beta;
    //! The location to perform the clustering in direction
    const double location;

   public:
    /**
     * Performs one point clustering
     * @param direction The direction (0, 1, 2) to perform the mapping
     * @param start The start of the domain in direction
     * @param end The end of the domain in direction
     * @param beta The clustering factor
     * @param location The location to perform the clustering in direction
     */
    explicit OnePointClusteringMapper(int direction, double start, double end, double beta, double location);

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
