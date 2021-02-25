#ifndef ABLATELIBRARY_BUILDER_HPP
#define ABLATELIBRARY_BUILDER_HPP
#include <memory>
#include "parser/factory.hpp"
namespace ablate {
class Builder {
   public:
    /**
     * default run method for particles and flow
     * @param factory
     */
    static void Run(std::shared_ptr<ablate::parser::Factory> factory);
};
}  // namespace ablate

#endif  // ABLATELIBRARY_BUILDER_HPP
