#ifndef ABLATELIBRARY_DOMAIN_INITIALIZERLIST_HPP
#define ABLATELIBRARY_DOMAIN_INITIALIZERLIST_HPP

#include <memory>
#include <utility>
#include <vector>
#include "field.hpp"
#include "initializer.hpp"
#include "mathFunctions/fieldFunction.hpp"

namespace ablate::domain {
/**
 * Simple class used to produce the field functions for initialization
 */
class InitializerList : public Initializer {
   private:
    const std::vector<std::shared_ptr<Initializer>> initializers;

   public:
    /**
     * Create a simple Initializer with a fixed set of fieldFunctions
     */
    explicit InitializerList(std::vector<std::shared_ptr<Initializer>>);

    /**
     * Interface to produce the field functions from fields
     */
    [[nodiscard]] std::vector<std::shared_ptr<mathFunctions::FieldFunction>> GetFieldFunctions(const std::vector<domain::Field>& fields) const override;
};

}  // namespace ablate::domain

#endif  // ABLATELIBRARY_DOMAIN_INITIALIZERLIST_HPP
