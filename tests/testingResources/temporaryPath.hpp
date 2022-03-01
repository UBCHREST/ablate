#ifndef ABLATELIBRARY_TEMPORARYPATH_HPP
#define ABLATELIBRARY_TEMPORARYPATH_HPP

#include <filesystem>

namespace testingResources {

class TemporaryPath {
   private:
    const std::filesystem::path path;

    //! list of chars used for random string
    inline const static std::string charSet ="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    //! used to create a random temp file name
    static std::string RandomString(std::size_t length);

   public:
    /**
     * Create a TemporaryPath that will be removed when TemporaryPath leave scopes
     */
    TemporaryPath();

    /**
     * clean up the temp file
     */
    ~TemporaryPath();

    /**
     * Return the path
     */
     inline const std::filesystem::path& GetPath() const{
         return path;
     }

     /**
     * Reads the file into a string
      */
      std::string ReadFile() const;
};

}

#endif  // ABLATELIBRARY_TEMPORARYPATH_HPP
