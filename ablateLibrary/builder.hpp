#ifndef ABLATELIBRARY_BUILDER_HPP
#define ABLATELIBRARY_BUILDER_HPP
#include "parser/factory.hpp"
#include <memory>
namespace ablate{
class Builder {
   public:
    /**
     * default run method for particles and flow
     * @param factory
     */
    static void Run(std::shared_ptr<ablate::parser::Factory> factory);
};
}

#endif  // ABLATELIBRARY_BUILDER_HPP
