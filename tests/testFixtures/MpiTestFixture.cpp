#include "MpiTestFixture.hpp"

#include <filesystem>
#include <fstream>

int* MpiTestFixture::argc;
char*** MpiTestFixture::argv;
const std::string MpiTestFixture::InTestRunFlag = "InMpiTestRun";
bool MpiTestFixture::inMpiTestRun;

bool MpiTestFixture::InitializeTestingEnvironment(int* argc, char***argv){
    MpiTestFixture::argc = argc;
    MpiTestFixture::argv = argv;

    int inMpiTestRunLocation = -1;
    for(auto i =0; i < *argc; i++){
        if(strcmp(MpiTestFixture::InTestRunFlag.c_str(), (*argv)[i]) == 0){
            inMpiTestRun = true;
            inMpiTestRunLocation = i;
        }
    }

    if(inMpiTestRunLocation >= 0){
        *argc = (*argc)-1;
        for(auto i = inMpiTestRunLocation; i < *argc; i++ ){
            (*argv)[i] = (*argv)[i+1];
        }
    }

    return inMpiTestRun;
}

void MpiTestFixture::SetUp() {
}

void MpiTestFixture::TearDown(){
    if(!inMpiTestRun){
        fs::remove_all(OutputFile());
    }
}

void MpiTestFixture::RunWithMPI() const {
    // build the mpi command
    std::stringstream mpiCommand;
    mpiCommand << "mpirun ";
    mpiCommand << "-n " << mpiTestParameter.nproc << " ";
    mpiCommand << ExecutablePath() << " ";
    mpiCommand << InTestRunFlag << " ";
    mpiCommand << "--gtest_filter=" << TestName() << " ";
    mpiCommand << mpiTestParameter.arguments << " ";
    mpiCommand << " > " << OutputFile();

    std::system(mpiCommand.str().c_str());
}

void MpiTestFixture::CompareOutputFiles(){
    // load the actual output
    std::ifstream actualStream(OutputFile());
    std::string actual((std::istreambuf_iterator<char>(actualStream)), std::istreambuf_iterator<char>());

    // read in the expected
    std::ifstream expectedStream(mpiTestParameter.expectedOutputFile);
    std::string expected((std::istreambuf_iterator<char>(expectedStream)), std::istreambuf_iterator<char>());

    ASSERT_TRUE(actual.length() > 0) << "Actual output is expected not to be empty";
    ASSERT_EQ(actual, expected);
}

std::ostream& operator<<(std::ostream& os, const MpiTestParameter& params){
    return os << params.expectedOutputFile;
}