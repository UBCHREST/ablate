#ifndef ABLATELIBRARY_DOMAIN_INITIALIZER_HPP
#define ABLATELIBRARY_DOMAIN_INITIALIZER_HPP

#include <memory>
#include <utility>
#include <vector>
#include "field.hpp"
#include "mathFunctions/fieldFunction.hpp"

namespace ablate::domain {
/**
 * Simple class used to produce the field functions for initialization
 */
class Initializer {
   private:
    const std::vector<std::shared_ptr<mathFunctions::FieldFunction>> fieldFunctions;

   public:
    /**
     * Create an empty list
     */
    Initializer() = default;

    /**
     * Create a simple Initializer with a fixed set of fieldFunctions
     */
    explicit Initializer(std::vector<std::shared_ptr<mathFunctions::FieldFunction>>);

    /**
     * Create a simple Initializer with a fixed set of fieldFunctions
     */
    template <class... FieldFunctions>
    explicit Initializer(FieldFunctions&&... functions) : fieldFunctions{std::forward<FieldFunctions>(functions)...} {};

    /**
     * Optional cleanup
     */
    virtual ~Initializer() = default;

    /**
     * Interface to produce the field functions from fields
     */
    [[nodiscard]] virtual std::vector<std::shared_ptr<mathFunctions::FieldFunction>> GetFieldFunctions(const std::vector<domain::Field>& fields) const { return fieldFunctions; }
};

}  // namespace ablate::domain

#endif  // ABLATELIBRARY_DOMAIN_INITIALIZER_HPP
