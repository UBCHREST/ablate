#ifndef ABLATELIBRARY_BUILDER_HPP
#define ABLATELIBRARY_BUILDER_HPP
#include <memory>
#include <ostream>
#include "parser/factory.hpp"

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

    /**
     * print the version information for the ablate library
     * @param stream
     */
    static void PrintInfo(std::ostream& stream);
};
}  // namespace ablate

#endif  // ABLATELIBRARY_BUILDER_HPP
