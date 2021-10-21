#ifndef ABLATELIBRARY_BOXINITIALIZER_HPP
#define ABLATELIBRARY_BOXINITIALIZER_HPP
#include "initializer.hpp"

namespace ablate::particles::initializers {
class BoxInitializer : public Initializer {
   private:
    const std::vector<double> lowerBound;
    const std::vector<double> upperBound;
    const int particlesPerDim;

   public:
    explicit BoxInitializer(std::vector<double> lowerBound = {0, 0, 0}, std::vector<double> upperBound = {1.0, 1.0, 1.0}, int particlesPerDim = 1);
    ~BoxInitializer() = default;

    void Initialize(ablate::domain::SubDomain& flow, DM particleDM) override;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_BOXINITIALIZER_HPP
