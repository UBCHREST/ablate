#ifndef ABLATELIBRARY_OUTPUT_DIRECTORY_HPP
#define ABLATELIBRARY_OUTPUT_DIRECTORY_HPP

#include "pathLocator.hpp"

namespace ablate::environment {
/**
 * Returns a path relative to the output directory
 */
class OutputDirectory : public cppParser::PathLocator {
   private:
    /**
     * the relative path to the file in the output directory
     */
    const std::string relativePath;

   public:
    /**
     * Provides a path to a file relative to the current output directory
     */
    explicit OutputDirectory(std::string relativePath);

    /**
     * Returns the absolute path given the current output directory
     * @param searchPaths
     * @return
     */
    std::filesystem::path Locate(const std::vector<std::filesystem::path>& searchPaths) override;
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_OUTPUT_DIRECTORY_HPP
