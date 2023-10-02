#ifndef ABLATELIBRARY_CHEMTABTESTFIXTURE_HPP
#define ABLATELIBRARY_CHEMTABTESTFIXTURE_HPP

#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <functional>
#include <map>
#include <ostream>
#include <string>
#include "eos/eos.hpp"
#include "gmock/gmock.h"
#include "petscTestFixture.hpp"

/*******************************************************************************************************
 * Tests for expected input/outputs
 */
struct ChemTabTestParameters {
    std::filesystem::path modelPath;
    std::string testTargetFile;
};
class ChemTabTestFixture : public testingResources::PetscTestFixture, public testing::WithParamInterface<ChemTabTestParameters> {
   protected:
    YAML::Node testTargets;

    void SetUp() override {
        testingResources::PetscTestFixture::SetUp();
        testTargets = YAML::LoadFile(GetParam().testTargetFile);

        // this should be an array
        if (!testTargets.IsSequence()) {
            FAIL() << "The provided test targets " + GetParam().testTargetFile + " must be an sequence.";
        }
    }
};

#endif  // ABLATELIBRARY_CHEMTABTESTFIXTURE_HPP
