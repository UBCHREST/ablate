#include "MpiTestFixture.h"
#include <filesystem>
#include <fstream>

int* MpiTestFixture::argc;
char*** MpiTestFixture::argv;
const std::string MpiTestFixture::InTestRunFlag = "InMpiTestRun";
bool MpiTestFixture::inMpiTestRun;

bool MpiTestFixture::InitializeTestingEnvironment(int* argc, char***argv){
    MpiTestFixture::argc = argc;
    MpiTestFixture::argv = argv;

    for(auto i =0; i < *MpiTestFixture::argc; i++){
        if(strcmp(MpiTestFixture::InTestRunFlag.c_str(), (*MpiTestFixture::argv)[i]) == 0){
            inMpiTestRun = true;
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
    mpiCommand << "-n " << GetParam().nproc << " ";
    mpiCommand << ExecutablePath() << " ";
    mpiCommand << InTestRunFlag << " ";
    mpiCommand << "--gtest_filter=" << TestName() << " ";
    mpiCommand << " >> " << OutputFile();

    std::system(mpiCommand.str().c_str());
}

void MpiTestFixture::CompareOutputFiles(){
    // load the actual output
    std::ifstream actualStream(OutputFile());
    std::string actual((std::istreambuf_iterator<char>(actualStream)), std::istreambuf_iterator<char>());

    // read in the expected
    std::ifstream expectedStream(GetParam().expectedOutputFile);
    std::string expected((std::istreambuf_iterator<char>(expectedStream)), std::istreambuf_iterator<char>());

    ASSERT_TRUE(actual.length() > 0) << "Actual output is expected not to be empty";
    ASSERT_EQ(actual, expected);
}

std::ostream& operator<<(std::ostream& os, const MpiTestParameter& params){
    return os << "MPI Params: " << params.nproc;  // whatever needed to print bar to os
}