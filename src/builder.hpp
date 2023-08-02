#ifndef ABLATELIBRARY_BUILDER_HPP
#define ABLATELIBRARY_BUILDER_HPP

#include <memory>
#include <ostream>
#include <solver/timeStepper.hpp>
#include "factory.hpp"

namespace ablate {
class Builder {
   public:
    /**
     * build the time stepper to run
     * @param factory
     */
    static std::shared_ptr<ablate::solver::TimeStepper> Build(const std::shared_ptr<cppParser::Factory>& factory);

    /**
     * default run method for particles and flow
     * @param factory
     */
    static void Run(const std::shared_ptr<cppParser::Factory>& factory);

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
