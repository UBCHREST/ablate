#ifndef ABLATELIBRARY_BUILDER_HPP
#define ABLATELIBRARY_BUILDER_HPP
#include <memory>
#include "parser/factory.hpp"
#include <ostream>

namespace ablate {
class Builder {
   public:
    /**
     * default run method for particles and flow
     * @param factory
     */
    static void Run(std::shared_ptr<ablate::parser::Factory> factory);

    /**
     * print the version information for the ablate library
     * @param stream
     */
    static void PrintVersion(std::ostream& stream);
};
}  // namespace ablate

#endif  // ABLATELIBRARY_BUILDER_HPP
