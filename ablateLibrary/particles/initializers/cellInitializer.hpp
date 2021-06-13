#ifndef ABLATELIBRARY_CELLINITIALIZER_HPP
#define ABLATELIBRARY_CELLINITIALIZER_HPP
#include "initializer.hpp"

namespace ablate::particles::initializers {
class CellInitializer : public Initializer {
   private:
    const int particlesPerCell;

   public:
    explicit CellInitializer(int particlesPerCellPerDim = 1);
    ~CellInitializer() override = default;

    void Initialize(ablate::flow::Flow& flow, DM particleDM) override;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_CELLINITIALIZER_HPP