#include "testRunEnvironment.hpp"
#include "environment/runEnvironment.hpp"
#include "parameters/mapParameters.hpp"

testingResources::TestRunEnvironment::TestRunEnvironment(std::string outputDir, std::string title, bool tagDirectory) {
    ablate::parameters::MapParameters parameters({{"directory", outputDir}, {"title", title}, {"tagDirectory", std::to_string(tagDirectory)}});
    ablate::environment::RunEnvironment::Setup(parameters);
}
testingResources::TestRunEnvironment::TestRunEnvironment(const ablate::parameters::Parameters& parameters, std::filesystem::path inputPath) {
    ablate::environment::RunEnvironment::Setup(parameters, inputPath);
}

testingResources::TestRunEnvironment::~TestRunEnvironment() { ablate::environment::RunEnvironment::Setup(); }
