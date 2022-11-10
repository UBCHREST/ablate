#include <memory>
#include "domain/modifiers/onePointClusteringMapper.hpp"
#include "mathFunctions/functionFactory.hpp"
#include "meshMapperTestFixture.hpp"
#include "utilities/vectorUtilities.hpp"

namespace ablateTesting::domain::modifier {

INSTANTIATE_TEST_SUITE_P(OnePointClusteringMapperTests, MeshMapperTestFixture,
                         testing::Values((MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::OnePointClusteringMapper>(0, 0.0, 3.3, 4.0, 1.1); },
                                                                    .testingValues = {TestingPair{.in = {0.1, 0.2}, .out = {0.1354546341805546, 0.2}},
                                                                                      TestingPair{.in = {0, -.1}, .out = {0.0, -.1}},
                                                                                      TestingPair{.in = {1.1, 0.825}, .out = {0.9536360483220486, 0.825}},
                                                                                      TestingPair{.in = {3.3, 1.55}, .out = {3.3, 1.55}},
                                                                                      TestingPair{.in = {3.0, -.05, -2}, .out = {2.6137852114073157, -.05, -2}}}},
                                         (MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::OnePointClusteringMapper>(1, -.1, 1.55, 3.0, 0.825); },
                                                                    .testingValues = {TestingPair{.in = {0.1, 0.2}, .out = {.1, 0.3329854477992793}},
                                                                                      TestingPair{.in = {0, -.1}, .out = {0.0, -.1}},
                                                                                      TestingPair{.in = {1.1, 0.825}, .out = {1.1, 0.8526209630423186}},
                                                                                      TestingPair{.in = {3.3, 1.55}, .out = {3.3, 1.55}},
                                                                                      TestingPair{.in = {3.0, -.05, -2}, .out = {3.0, -0.01261145487549653, -2}}}},
                                         (MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::OnePointClusteringMapper>(2, -.1, 1.55, 3.0, 0.825); },
                                                                    .testingValues = {TestingPair{.in = {0.0, 0.1, .2}, .out = {0.0, 0.1, 0.3329854477992793}}}}),
                         [](const testing::TestParamInfo<MeshMapperTestParameters>& info) { return std::to_string(info.index); });

}  // namespace ablateTesting::domain::modifier