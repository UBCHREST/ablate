#ifndef ABLATELIBRARY_BOXINITIALIZER_HPP
#define ABLATELIBRARY_BOXINITIALIZER_HPP
#include "initializer.hpp"

namespace ablate::particles::initializers {
class BoxInitializer : public Initializer {
   public:
    explicit BoxInitializer(std::map<std::string, std::string> arguments);
    ~BoxInitializer() = default;

    void Initialize(ablate::flow::Flow& flow, DM particleDM) override;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_BOXINITIALIZER_HPP
