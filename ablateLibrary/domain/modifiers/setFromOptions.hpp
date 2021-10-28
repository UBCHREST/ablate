#ifndef ABLATELIBRARY_SETFROMOPTIONS_HPP
#define ABLATELIBRARY_SETFROMOPTIONS_HPP

#include <parameters/parameters.hpp>
#include "modifier.hpp"

namespace ablate::domain::modifier {

class SetFromOptions : public Modifier {
   private:
    PetscOptions petscOptions;

   public:
    explicit SetFromOptions(std::shared_ptr<parameters::Parameters> options = {});
    ~SetFromOptions() override;

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifier
#endif  // ABLATELIBRARY_SETFROMOPTIONS_HPP
