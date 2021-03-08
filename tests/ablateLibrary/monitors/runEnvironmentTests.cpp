#include <fstream>
#include <memory>
#include <sstream>
#include "gtest/gtest.h"
#include "monitors/runEnvironment.hpp"
#include <stdlib.h>
#include <time.h>
#include <string>
#include "parameters/mockParameters.hpp"

namespace ablateTesting::monitors {

using namespace ablate::monitors;

class RunEnvironmentTestFixture: public ::testing::Test {
   public:
    std::filesystem::path tempInputFile;
    std::string uniqueTitle;

    void SetUp() {
        // create a dummy file
        tempInputFile = std::filesystem::temp_directory_path();
        tempInputFile /= "tempFile.yaml";
        std::ofstream ofs(tempInputFile);
        ofs << "---" << std::endl;
        ofs << " item: \"im a string!\"";
        ofs.close();

        // create a title
        srand( (unsigned)time(NULL) );
        uniqueTitle = "title_" + std::to_string(rand());
    }

    void TearDown() {
        std::filesystem::remove_all(tempInputFile);
    }
};

TEST_F(RunEnvironmentTestFixture, ShouldSetupDefaultEnviroment) {
    // arrange
    // setup the mock parameters
    ablateTesting::parameters::MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("title")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(uniqueTitle));
    EXPECT_CALL(mockParameters, GetString("tagDirectory")).Times(::testing::Exactly(1));
    EXPECT_CALL(mockParameters, GetString("outputDirectory")).Times(::testing::Exactly(1));

    // act
    ablate::monitors::RunEnvironment runEnvironment(tempInputFile, mockParameters);

    // assert
    ASSERT_TRUE(runEnvironment.GetOutputDirectory().filename().string().rfind(uniqueTitle, 0) == 0) << "the output directory should start with the title";
    ASSERT_EQ(tempInputFile.parent_path(), runEnvironment.GetOutputDirectory().parent_path()) << "the output directory should be next to the input file";
    ASSERT_GT(runEnvironment.GetOutputDirectory().string().length(), uniqueTitle.length()) << "the output directory include additional date/time/info";
    ASSERT_TRUE(std::filesystem::exists(runEnvironment.GetOutputDirectory()/"tempFile.yaml")) << "the output directory should contain a copy of the input file";
    ASSERT_EQ(std::filesystem::file_size(runEnvironment.GetOutputDirectory()/"tempFile.yaml"), std::filesystem::file_size(tempInputFile) ) << "the copied input file should be the same size";

    // cleanup
    std::filesystem::remove_all(runEnvironment.GetOutputDirectory());
}

TEST_F(RunEnvironmentTestFixture, ShouldNotTagOutputDirectory) {
    // arrange
    // setup the mock parameters
    ablateTesting::parameters::MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("title")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(uniqueTitle));
    EXPECT_CALL(mockParameters, GetString("tagDirectory")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("false"));
    EXPECT_CALL(mockParameters, GetString("outputDirectory")).Times(::testing::Exactly(1));

    // act
    ablate::monitors::RunEnvironment runEnvironment(tempInputFile, mockParameters);

    // assert
    ASSERT_EQ(runEnvironment.GetOutputDirectory().filename().string(), uniqueTitle) << "the output directory should be the title";
    ASSERT_EQ(tempInputFile.parent_path(), runEnvironment.GetOutputDirectory().parent_path()) << "the output directory should be next to the input file";
    ASSERT_GT(runEnvironment.GetOutputDirectory().string().length(), uniqueTitle.length()) << "the output directory include additional date/time/info";
    ASSERT_TRUE(std::filesystem::exists(runEnvironment.GetOutputDirectory()/"tempFile.yaml")) << "the output directory should contain a copy of the input file";
    ASSERT_EQ(std::filesystem::file_size(runEnvironment.GetOutputDirectory()/"tempFile.yaml"), std::filesystem::file_size(tempInputFile) ) << "the copied input file should be the same size";

    // cleanup
    std::filesystem::remove_all(runEnvironment.GetOutputDirectory());
}

TEST_F(RunEnvironmentTestFixture, ShouldUseAndTagSpecifiedOutputDirectory) {
    // arrange
    auto outputDirectory = std::filesystem::temp_directory_path()/ ("specified_output_dir_" + std::to_string(rand()));

    // setup the mock parameters
    ablateTesting::parameters::MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("title")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(uniqueTitle));
    EXPECT_CALL(mockParameters, GetString("tagDirectory")).Times(::testing::Exactly(1));
    EXPECT_CALL(mockParameters, GetString("outputDirectory")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(outputDirectory.string()));

    // act
    ablate::monitors::RunEnvironment runEnvironment(tempInputFile, mockParameters);

    // assert
    ASSERT_TRUE(runEnvironment.GetOutputDirectory().filename().string().rfind(outputDirectory.filename().string(), 0) == 0) << "the output directory should start with specified_output_dir_";
    ASSERT_EQ(outputDirectory.parent_path(), runEnvironment.GetOutputDirectory().parent_path()) << "the output directory should be in the specified location";
    ASSERT_GT(runEnvironment.GetOutputDirectory().string().length(), uniqueTitle.length()) << "the output directory include additional date/time/info";
    ASSERT_TRUE(std::filesystem::exists(runEnvironment.GetOutputDirectory()/"tempFile.yaml")) << "the output directory should contain a copy of the input file";
    ASSERT_EQ(std::filesystem::file_size(runEnvironment.GetOutputDirectory()/"tempFile.yaml"), std::filesystem::file_size(tempInputFile) ) << "the copied input file should be the same size";

    // cleanup
    std::filesystem::remove_all(runEnvironment.GetOutputDirectory());
}

TEST_F(RunEnvironmentTestFixture, ShouldUseWithoutTaggingSpecifiedOutputDirectory) {
    // arrange
    auto outputDirectory = std::filesystem::temp_directory_path()/ ("specified_output_dir_" + std::to_string(rand()));

    // setup the mock parameters
    ablateTesting::parameters::MockParameters mockParameters;
    EXPECT_CALL(mockParameters, GetString("title")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(uniqueTitle));
    EXPECT_CALL(mockParameters, GetString("tagDirectory")).Times(::testing::Exactly(1)).WillOnce(::testing::Return("false"));
    EXPECT_CALL(mockParameters, GetString("outputDirectory")).Times(::testing::Exactly(1)).WillOnce(::testing::Return(outputDirectory.string()));

    // act
    ablate::monitors::RunEnvironment runEnvironment(tempInputFile, mockParameters);

    // assert
    ASSERT_EQ(outputDirectory, runEnvironment.GetOutputDirectory()) << "the output directory should the one specified";
    ASSERT_GT(runEnvironment.GetOutputDirectory().string().length(), uniqueTitle.length()) << "the output directory include additional date/time/info";
    ASSERT_TRUE(std::filesystem::exists(runEnvironment.GetOutputDirectory()/"tempFile.yaml")) << "the output directory should contain a copy of the input file";
    ASSERT_EQ(std::filesystem::file_size(runEnvironment.GetOutputDirectory()/"tempFile.yaml"), std::filesystem::file_size(tempInputFile) ) << "the copied input file should be the same size";

    // cleanup
    std::filesystem::remove_all(runEnvironment.GetOutputDirectory());
}

}  // namespace ablateTesting::parser