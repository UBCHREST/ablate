#ifndef ABLATELIBRARY_SETFROMOPTIONS_HPP
#define ABLATELIBRARY_SETFROMOPTIONS_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include "modifier.hpp"

namespace ablate::domain::modifiers {

class SetFromOptions : public Modifier {
   private:
    PetscOptions petscOptions;

   public:
    explicit SetFromOptions(std::shared_ptr<parameters::Parameters> options = {});
    ~SetFromOptions() override;

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_SETFROMOPTIONS_HPP
