#ifndef ABLATELIBRARY_MESHMAPPERTESTFIXTURE_HPP
#define ABLATELIBRARY_MESHMAPPERTESTFIXTURE_HPP
#include <functional>
#include <memory>
#include "domain/modifiers/meshMapper.hpp"
#include "petscTestFixture.hpp"

namespace ablateTesting::domain::modifier {

struct TestingPair {
    std::vector<double> in;
    std::vector<double> out;
};

struct MeshMapperTestParameters {
    /**
     * Function to create the test mapper
     */
    std::function<std::shared_ptr<ablate::domain::modifiers::MeshMapper>()> createMapper;

    /**
     * vector of input/expected output parameters
     */
    std::vector<TestingPair> testingValues;
};

class MeshMapperTestFixture : public testingResources::PetscTestFixture, public ::testing::WithParamInterface<MeshMapperTestParameters> {};

}  // namespace ablateTesting::domain::modifier
#endif  // ABLATELIBRARY_MESHMAPPERTESTFIXTURE_HPP
