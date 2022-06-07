#ifndef ABLATELIBRARY_DOWNLOAD_HPP
#define ABLATELIBRARY_DOWNLOAD_HPP

#include <array>
#include "pathLocator.hpp"
namespace ablate::environment {
class Download : public cppParser::PathLocator {
   private:
    /**
     * url to download file from
     */
    const std::string url;

    /**
     * keep list of urlPrefixes
     */
    constexpr static std::array<std::string_view, 4> urlPrefixes = {"https://", "http://", "file://", "https://"};

   public:
    /**
     * downloads the specified file.  Relocates if temporaryFile if false
     * @param url
     * @param temporaryFile
     */
    explicit Download(std::string url);

    /**
     * Downloads and relocates the file
     * @param searchPaths
     * @return
     */
    std::filesystem::path Locate(const std::vector<std::filesystem::path>& searchPaths = {}) override;

    /**
     * simple helper function to determine if the provided string is a url
     * @return
     */
    static bool IsUrl(const std::string& testUrl) {
        return std::any_of(urlPrefixes.begin(), urlPrefixes.end(), [&testUrl](const auto& prefix) { return testUrl.rfind(prefix, 0) == 0; });
    }
};
}  // namespace ablate::environment

#endif  // ABLATELIBRARY_DOWNLOAD_HPP
