#ifndef mpitestfixture_h
#define mpitestfixture_h
#include <gtest/gtest.h>
#include <filesystem>
namespace fs = std::filesystem;

struct MpiTestParameter
{
    int nproc;
    std::string expectedOutputFile;
};

class MpiTestFixture : public ::testing::TestWithParam<MpiTestParameter> {
protected:
    static int* argc;
    static char*** argv;
    static bool inMpiTestRun;
    static const std::string InTestRunFlag;

    void SetUp() override;

    void TearDown() override;

    void RunWithMPI() const;

    void CompareOutputFiles();

    std::string ExecutablePath() const{
        return std::string(*argv[0]);
    }

    std::string TestName() const{
        return std::string(::testing::UnitTest::GetInstance()->current_test_info()->test_suite_name()) + "." + std::string(::testing::UnitTest::GetInstance()->current_test_info()->name());
    }

    std::string OutputFile() const{
        auto fileName = TestName() + ".txt";
        std::replace(fileName.begin(), fileName.end(), '/', '_');
        return fileName;
    }

public:
    static bool InitializeTestingEnvironment(int* argc, char***argv);
};

std::ostream& operator<<(std::ostream& os, const MpiTestParameter& params);

// Define macros to simplify the setup and running of mpi based code
#define StartWithMPI if(inMpiTestRun){
#define EndWithMPI  }else{RunWithMPI();CompareOutputFiles();}

#endif //mpitestfixture_h
