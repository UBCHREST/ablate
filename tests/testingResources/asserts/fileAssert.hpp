#ifndef TESTING_RESOURCE_FILEASSERT
#define TESTING_RESOURCE_FILEASSERT
#include <filesystem>
#include <memory>
namespace testingResources::asserts {

/**
 * This is a helper mixin that can be used to compare files with expected reference files
 */
class FileAssert {
   public:
    /**
     * Compares the expected file and the actual file and calls fail if they do not match
     * @param expectedFileName
     * @param actualFileName
     */
    static void CompareFile(const std::filesystem::path& expectedFileName, const std::filesystem::path& actualFileName);
};

}  // namespace testingResources::asserts

#endif  // TESTING_RESOURCE_POSTRUNASSERT