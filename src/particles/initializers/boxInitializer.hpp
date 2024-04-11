#ifndef ABLATELIBRARY_BOXINITIALIZER_HPP
#define ABLATELIBRARY_BOXINITIALIZER_HPP
#include "initializer.hpp"

namespace ablate::particles::initializers {
/**
 * Setup particles over a set box
 */
class BoxInitializer : public Initializer {
   private:
    const std::vector<double> lowerBound;
    const std::vector<double> upperBound;
    const int particlesPerDim;

   public:
    /**
     * Insert particles over a set box
     * @param lowerBound the lower bound of the box
     * @param upperBound the upper bound of the box
     * @param particlesPerDim particles per dimension in the box
     */
    explicit BoxInitializer(std::vector<double> lowerBound = {0, 0, 0}, std::vector<double> upperBound = {1.0, 1.0, 1.0}, int particlesPerDim = 1);
    ~BoxInitializer() override = default;

    /**
     * initialize over the box
     * @param flow
     * @param particleDM
     */
    void Initialize(ablate::domain::SubDomain& flow, DM particleDM) override;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_BOXINITIALIZER_HPP
