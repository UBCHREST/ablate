#include <PetscTestFixture.hpp>
#include <fstream>
#include <memory>
#include <parameters/mapParameters.hpp>
#include <sstream>
#include <utilities/fileUtility.hpp>
#include "environment/runEnvironment.hpp"
#include "gtest/gtest.h"
#include "yamlParser.hpp"

namespace ablateTesting::parser {

#define REMOTE_URL "https://raw.githubusercontent.com/UBCHREST/ablate/main/tests/ablateLibrary/inputs/eos/thermo30.dat"

namespace fs = std::filesystem;

TEST(YamlParserTests, ShouldLocateLocalFile) {
    // arrange
    fs::path tmpFile = fs::temp_directory_path() / "tempFile.txt";
    {
        std::ofstream ofs(tmpFile);
        ofs << " tempFile" << std::endl;
        ofs.close();
    }

    fs::path tempYaml = fs::temp_directory_path();
    tempYaml /= "tempFile.yaml";
    {
        std::ofstream ofs(tempYaml);
        ofs << "---" << std::endl;
        ofs << " fileName: " << tmpFile << std::endl;
        ofs.close();
    }

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {tempYaml.parent_path()});
    auto yamlParser = std::make_shared<cppParser::YamlParser>(tempYaml, fileLocator.GetLocateFileFunction());

    // act
    auto computedFilePath = yamlParser->Get(cppParser::ArgumentIdentifier<std::filesystem::path>{"fileName"});

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, tmpFile);

    // cleanup
    fs::remove(tmpFile);
    fs::remove(tempYaml);
}

TEST(YamlParserTests, ShouldLocateFileNextToInputFile) {
    // arrange
    fs::path tempYaml = fs::temp_directory_path();
    tempYaml /= "tempFile.yaml";
    {
        std::ofstream ofs(tempYaml);
        ofs << "---" << std::endl;
        ofs << " mesh: " << std::endl;
        ofs << "   fileName: tempFileNameForTesting.txt" << std::endl;
        ofs.close();
    }

    fs::path tmpFile = tempYaml.parent_path() / "tempFileNameForTesting.txt";
    {
        std::ofstream ofs(tmpFile);
        ofs << " tempFile" << std::endl;
        ofs.close();
    }

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {tempYaml.parent_path()});
    auto yamlParser = std::make_shared<cppParser::YamlParser>(tempYaml, fileLocator.GetLocateFileFunction());
    auto yamlMeshFactory = yamlParser->GetFactory("mesh");

    // act
    auto computedFilePath = yamlMeshFactory->Get(cppParser::ArgumentIdentifier<std::filesystem::path>{"fileName"});

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(computedFilePath, std::filesystem::canonical(tmpFile));
    ASSERT_EQ(tempYaml.parent_path(), tempYaml.parent_path());

    // cleanup
    fs::remove(tmpFile);
    fs::remove(tempYaml);
}

class YamlParserTestsPetscTestFixture : public testingResources::PetscTestFixture {};

TEST_F(YamlParserTestsPetscTestFixture, ShouldDownloadFile) {
    // arrange
    fs::path tempYaml = fs::temp_directory_path();
    tempYaml /= "tempFile.yaml";
    {
        std::ofstream ofs(tempYaml);
        ofs << "---" << std::endl;
        ofs << " fileName: " << REMOTE_URL << std::endl;
        ofs.close();
    }

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {tempYaml.parent_path()});
    auto yamlParser = std::make_shared<cppParser::YamlParser>(tempYaml, fileLocator.GetLocateFileFunction());

    // act
    auto computedFilePath = yamlParser->Get(cppParser::ArgumentIdentifier<std::filesystem::path>{"fileName"});

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));

    // cleanup
    fs::remove(computedFilePath);
    fs::remove(tempYaml);
}

TEST_F(YamlParserTestsPetscTestFixture, ShouldDownloadAndRelocateFile) {
    // arrange
    fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";

    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << "environment:" << std::endl;
    yaml << "  directory: " << outputDir << std::endl;
    yaml << "  title: test " << std::endl;
    yaml << "  tagDirectory: false" << std::endl;
    yaml << "fileName: " << REMOTE_URL << std::endl;

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF, {}, outputDir);
    auto yamlParser = std::make_shared<cppParser::YamlParser>(yaml.str(), fileLocator.GetLocateFileFunction());

    // Set the global environment
    auto params = yamlParser->GetByName<ablate::parameters::Parameters>("environment");
    ablate::environment::RunEnvironment::Setup(*params);

    // act
    auto computedFilePath = yamlParser->Get(cppParser::ArgumentIdentifier<std::filesystem::path>{"fileName"});

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_EQ(outputDir, computedFilePath.parent_path());

    // cleanup
    fs::remove(computedFilePath);
    fs::remove_all(outputDir);
}

TEST_F(YamlParserTestsPetscTestFixture, ShouldNotDownloadAndRelocateFile) {
    // arrange
    fs::path outputDir = fs::temp_directory_path() / "outputDirTemp";

    std::stringstream yaml;
    yaml << "---" << std::endl;
    yaml << "environment:" << std::endl;
    yaml << "  directory: " << outputDir << std::endl;
    yaml << "  title: test " << std::endl;
    yaml << "  tagDirectory: false" << std::endl;
    yaml << "fileName: " << REMOTE_URL << std::endl;

    ablate::utilities::FileUtility fileLocator(MPI_COMM_SELF);
    auto yamlParser = std::make_shared<cppParser::YamlParser>(yaml.str(), fileLocator.GetLocateFileFunction());

    // Set the global environment
    auto params = yamlParser->GetByName<ablate::parameters::Parameters>("environment");
    ablate::environment::RunEnvironment::Setup(*params);

    // act
    auto computedFilePath = yamlParser->Get(cppParser::ArgumentIdentifier<std::filesystem::path>{"fileName"});

    // assert
    ASSERT_TRUE(std::filesystem::exists(computedFilePath));
    ASSERT_NE(outputDir, computedFilePath.parent_path());

    // cleanup
    fs::remove(computedFilePath);
    fs::remove_all(outputDir);
}

}  // namespace ablateTesting::parser
