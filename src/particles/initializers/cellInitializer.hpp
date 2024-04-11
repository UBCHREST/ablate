#ifndef ABLATELIBRARY_CELLINITIALIZER_HPP
#define ABLATELIBRARY_CELLINITIALIZER_HPP
#include "initializer.hpp"

namespace ablate::particles::initializers {
/**
 * An Initializer that inserts a set number of particles per cell
 */
class CellInitializer : public Initializer {
   private:
    const int particlesPerCell;

   public:
    /**
     * An Initializer that inserts a set number of particles per cell
     * @param particlesPerCellPerDim inserts a set number of particles per dim
     */
    explicit CellInitializer(int particlesPerCellPerDim = 1);
    ~CellInitializer() override = default;

    /**
     * Add the particles to the particle DM
     * @param flow
     * @param particleDM
     */
    void Initialize(ablate::domain::SubDomain& flow, DM particleDM) override;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_CELLINITIALIZER_HPP