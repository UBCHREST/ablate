#ifndef mpitestfixture_h
#define mpitestfixture_h
#include <gtest/gtest.h>
#include <petscsys.h>
#include <filesystem>
#include "PetscTestErrorChecker.hpp"

namespace fs = std::filesystem;

namespace testingResources {

struct MpiTestParameter {
    std::string testName;
    int nproc;
    std::string expectedOutputFile;
    std::string arguments;

    std::string getTestName() const {
        std::string s = testName;
        std::replace(s.begin(), s.end(), ' ', '_');
        std::replace(s.begin(), s.end(), '/', '_');
        std::replace(s.begin(), s.end(), '.', '_');
        return s;
    }
};

class MpiTestFixture : public ::testing::Test {
   private:
    static bool inMpiTestRun;
    static bool keepOutputFile;
    static std::string mpiCommand;
    static std::string ParseCommandLineArgument(int* argc, char*** argv, const std::string flag);
    MpiTestParameter mpiTestParameter;

   protected:
    static int* argc;
    static char*** argv;
    static const std::string InTestRunFlag;
    static const std::string Test_Mpi_Command_Name;
    static const std::string Keep_Output_File;
    PetscTestErrorChecker testErrorChecker;

    void SetUp() override;

    void TearDown() override;

    void RunWithMPI() const;

    void CompareOutputFiles();

    void SetMpiParameters(MpiTestParameter mpiTestParameterIn) { mpiTestParameter = mpiTestParameterIn; }

    std::string ExecutablePath() const { return std::string(*argv[0]); }

    std::string TestName() const {
        return std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) + "." + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
    }

    std::string OutputFile() const {
        auto fileName = TestName() + ".txt";
        std::replace(fileName.begin(), fileName.end(), '/', '_');
        return fileName;
    }

    bool ShouldRunMpiCode() const { return inMpiTestRun || mpiTestParameter.nproc == 0; }

    template <typename T>
    std::string PrintVector(std::vector<T> values, const char* format) const {
        if (values.empty()) {
            return "[]";
        }

        char buff[100];
        std::snprintf(buff, sizeof(buff), format, values[0]);

        std::string result = "[" + std::string(buff);
        for (std::size_t i = 1; i < values.size(); i++) {
            std::snprintf(buff, sizeof(buff), format, values[i]);
            result += ", " + std::string(buff);
        }
        result += "]";
        return result;
    }

    std::filesystem::path MakeTemporaryPath(std::string name) const {
        auto path = std::filesystem::temp_directory_path() / name;
        if (std::filesystem::exists(path)) {
            std::filesystem::remove_all(path);
        }
        return path;
    }

    std::filesystem::path MakeTemporaryPath(std::string dir, std::string name) const {
        auto path = std::filesystem::temp_directory_path() / dir / name;
        if (std::filesystem::exists(path)) {
            std::filesystem::remove_all(path);
        }
        return path;
    }

   public:
    static bool InitializeTestingEnvironment(int* argc, char*** argv);
};

// Define macros to simplify the setup and running of mpi based code
#define StartWithMPI if (ShouldRunMpiCode()) {
#define EndWithMPI            \
    }                         \
    else {                    \
        RunWithMPI();         \
        CompareOutputFiles(); \
    }

}  // namespace testingResources
#endif  // mpitestfixture_h