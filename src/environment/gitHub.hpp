#ifndef ABLATELIBRARY_GITHUB_HPP
#define ABLATELIBRARY_GITHUB_HPP

#include "pathLocator.hpp"

namespace ablate::environment {
class GitHub : public cppParser::PathLocator {
   private:
    /**
     * repository to download file from (this includes owner/name)
     */
    const std::string repository;

    /**
     * the relative path in the repo
     */
    const std::string path;

    /**
     * github personal access token
     */
    const std::string gitHubToken;

    /**
     * hold constant values
     */
    const inline static std::string urlBase = "https://api.github.com/repos/";
    const inline static std::string contentSeperator = "/contents/";

    /**
     * private function to help download each file
     * @param urlPath
     * @param localPath
     */
    std::filesystem::path Download(std::string urlPath, std::filesystem::path localPath, bool root);

   public:
    /**
     * downloads the specified file.  Relocates if temporaryFile if false
     * @param repository
     * @param path
     * @param gitHubToken optional github [personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token) to access repo or
     * bypass rate limits.
     */
    explicit GitHub(std::string repository, std::string path, std::string gitHubToken = {});

    /**
     * Downloads and relocates the file
     * @param searchPaths
     * @return
     */
    std::filesystem::path Locate(const std::vector<std::filesystem::path>& searchPaths = {}) override;
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_DOWNLOAD_HPP
