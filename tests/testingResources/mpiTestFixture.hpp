#ifndef mpitestfixture_h
#define mpitestfixture_h
#include <gtest/gtest.h>
#include <petscsys.h>
#include <filesystem>
#include <utility>
#include "asserts/assert.hpp"
#include "petscTestErrorChecker.hpp"
#include "utilities/vectorUtilities.hpp"

namespace fs = std::filesystem;

namespace testingResources {

struct MpiTestParameter {
    // A general test name
    std::string testName;

    // The number of mpi processes to test against
    int nproc;

    // Optional program arguments
    std::string arguments;

    // Add options for ASAN flags
    std::string environment;

    // Keep a list of assert objects that can be used to determine if the case passed or failed
    std::vector<std::shared_ptr<asserts::Assert>> asserts;

    /**
     * base constructor that takes a single assert or list
     * @param testName
     * @param nproc
     * @param arguments
     * @param environment, optional environment flags
     */
    explicit MpiTestParameter(std::string testName, int nprocIn, std::string arguments, std::string environment, std::shared_ptr<asserts::Assert> assert,
                              const std::vector<std::shared_ptr<asserts::Assert>>& assertsIn)
        : testName(std::move(testName)),
          nproc(nprocIn > 0 ? nprocIn : 1),
          arguments(std::move(arguments)),
          environment(std::move(environment)),
          asserts(assert ? ablate::utilities::VectorUtilities::Merge({assert}, assertsIn) : assertsIn) {}

    /**
     * helper constructor for the mpi test parameters that takes a single of assert
     * @param testName
     * @param nproc
     * @param arguments
     * @param environment, optional environment flags
     */
    explicit MpiTestParameter(std::string testName, int nprocIn, std::string arguments, std::shared_ptr<asserts::Assert> assert, std::string environment = {})
        : testName(std::move(testName)), nproc(nprocIn > 0 ? nprocIn : 1), arguments(std::move(arguments)), environment(std::move(environment)), asserts({assert}) {}

    /**
     * helper constructor for the mpi test parameters that takes a list of asserts
     * @param testName
     * @param nproc
     * @param arguments
     * @param environment, optional environment flags
     */
    explicit MpiTestParameter(std::string testName = "", int nprocIn = 1, std::string arguments = "", std::vector<std::shared_ptr<asserts::Assert>> asserts = {}, std::string environment = {})
        : testName(std::move(testName)), nproc(nprocIn > 0 ? nprocIn : 1), arguments(std::move(arguments)), environment(std::move(environment)), asserts(std::move(asserts)) {}

    // A sanitized version of the test name
    [[nodiscard]] std::string getTestName() const {
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
    static std::string ParseCommandLineArgument(int* argc, char*** argv, std::string flag);
    /**
     * Keep a copy of the mpiTest parameter
     */
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

    /**
     * protected call to relaunch the executable under mpi
     */
    void RunWithMPI() const;

    /**
     * Setup the test parameter with the supplied mpiTestParameter from the test
     * @param mpiTestParameterIn
     */
    void SetMpiParameters(MpiTestParameter mpiTestParameterIn) { mpiTestParameter = std::move(mpiTestParameterIn); }

    [[nodiscard]] static std::string ExecutablePath() { return {*argv[0]}; }

    /**
     * The test from from the google test framework
     * @return
     */
    [[nodiscard]] std::string TestName() const {
        return std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) + "." + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
    }

    /**
     * Build the result directory before the run
     * @return
     */
    std::filesystem::path BuildResultDirectory() const;

    /**
     * Determine if we should run mpi code or launch a new processes instead
     * @return
     */
    [[nodiscard]] bool ShouldRunMpiCode() const { return inMpiTestRun || mpiTestParameter.nproc == 0; }

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

    [[nodiscard]] static std::filesystem::path MakeTemporaryPath(const std::string& name, MPI_Comm comm = MPI_COMM_SELF) {
        PetscMPIInt rank = 0;
        MPI_Comm_rank(comm, &rank);

        auto path = std::filesystem::temp_directory_path() / name;
        if (rank == 0) {
            if (std::filesystem::exists(path)) {
                std::filesystem::remove_all(path);
            }
        }

        MPI_Barrier(comm);
        return path;
    }

    [[nodiscard]] static std::filesystem::path MakeTemporaryPath(const std::string& dir, std::string name, MPI_Comm comm = MPI_COMM_SELF) {
        PetscMPIInt rank = 0;
        MPI_Comm_rank(comm, &rank);

        auto path = std::filesystem::temp_directory_path() / dir / name;
        if (std::filesystem::exists(path)) {
            std::filesystem::remove_all(path);
        }
        MPI_Barrier(comm);
        return path;
    }

    /**
     * Checks the asserts in the mpi test parameter
     */
    void CheckAsserts();

   public:
    static bool InitializeTestingEnvironment(int* argc, char*** argv);

    /**
     * Provide access to the output file path
     * @return
     */
    [[nodiscard]] std::string OutputFile() const {
        auto fileName = TestName() + ".txt";
        std::replace(fileName.begin(), fileName.end(), '/', '_');
        return fileName;
    }

    /**
     * Path to the result directory for the current test
     * @return
     */
    [[nodiscard]] std::filesystem::path ResultDirectory() const { return std::filesystem::current_path() / mpiTestParameter.getTestName(); }
};

// Define macros to simplify the setup and running of mpi based code
#define StartWithMPI if (ShouldRunMpiCode()) {
#define EndWithMPI      \
    }                   \
    else {              \
        RunWithMPI();   \
        CheckAsserts(); \
    }

}  // namespace testingResources

#endif  // mpitestfixture_h
