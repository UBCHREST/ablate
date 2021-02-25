#ifndef ABLATELIBRARY_CELLINITIALIZER_HPP
#define ABLATELIBRARY_CELLINITIALIZER_HPP
#include "initializer.hpp"

namespace ablate::particles::initializers {
class CellInitializer : public Initializer {
   public:
    explicit CellInitializer(std::map<std::string, std::string> arguments);
    ~CellInitializer() override = default;

    void Initialize(ablate::flow::Flow& flow, DM particleDM) override;
};
}  // namespace ablate::particles::initializers

#endif  // ABLATELIBRARY_CELLINITIALIZER_HPP