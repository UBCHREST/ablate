#ifndef ABLATELIBRARY_STRINGUTILITIES_HPP
#define ABLATELIBRARY_STRINGUTILITIES_HPP
#include <algorithm>
#include <string>
#include <string_view>

namespace ablate::utilities {
class StringUtilities {
   public:
    /**
     * Converts current string to upper case
     * @param str
     */
    static inline void ToUpper(std::string& str) { std::transform(str.begin(), str.end(), str.begin(), ::toupper); }

    /**
     * Converts current string to lower case
     * @param str
     */
    static inline void ToLower(std::string& str) { std::transform(str.begin(), str.end(), str.begin(), ::tolower); }

    /**
     * Converts current string to upper case
     * @param str
     */
    static inline std::string ToUpperCopy(const std::string_view& str) {
        std::string strcopy(str.size(), 0);
        std::transform(str.begin(), str.end(), strcopy.begin(), ::toupper);
        return strcopy;
    }

    /**
     * Converts current string to lower case
     * @param str
     */
    static inline std::string ToLowerCopy(const std::string_view& str) {
        std::string strcopy(str.size(), 0);
        std::transform(str.begin(), str.end(), strcopy.begin(), ::tolower);
        return strcopy;
    }

    /**
     * Check to see if subStr is in the str
     * @param str
     * @param substr
     * @return
     */
    static inline bool Contains(const std::string_view& str, const std::string_view& subTtr) { return str.find(subTtr) != str.npos; }

    /**
     * Check if the string ends with a suffix
     * @param str
     * @param suffix
     * @return
     */
    static bool EndsWith(std::string_view str, std::string_view suffix) { return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix); }

    /**
     * Check if a string starts with a prefix
     * @param str
     * @param prefix
     * @return
     */
    static bool StartsWith(std::string_view str, std::string_view prefix) { return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix); }

   private:
    StringUtilities() = delete;
};

}  // namespace ablate::utilities

#endif  // ABLATELIBRARY_MATHUTILITIES_HPP
