#include <memory>
#include "domain/modifiers/edgeClusteringMapper.hpp"
#include "mathFunctions/functionFactory.hpp"
#include "meshMapperTestFixture.hpp"
#include "utilities/vectorUtilities.hpp"

namespace ablateTesting::domain::modifier {

INSTANTIATE_TEST_SUITE_P(EdgeClusteringMapperTests, MeshMapperTestFixture,
                         testing::Values((MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::EdgeClusteringMapper>(0, 0.0, 3.3, 4.0); },
                                                                    .testingValues = {TestingPair{.in = {0.1, 0.2}, .out = {0.0959635725429939, 0.2}},
                                                                                      TestingPair{.in = {0, -.1}, .out = {0.0, -.1}},
                                                                                      TestingPair{.in = {1.1, 0.825}, .out = { 1.0738406097998698, 0.825}},
                                                                                      TestingPair{.in = {3.3, 1.55}, .out = {3.3, 1.55}},
                                                                                      TestingPair{.in = {3.0, -.05, -2}, .out = { 2.9935596949709646, -.05, -2}}

                                                                    }},
                                         (MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::EdgeClusteringMapper>(1, -.1, 1.55, 3.0); },
                                                                    .testingValues = {TestingPair{.in = {0.1, 0.2}, .out = {.1, 0.1828252256954647}},
                                                                                      TestingPair{.in = {0, -.1}, .out = {0.0, -.1}},
                                                                                      TestingPair{.in = {1.1, 0.825}, .out = {1.1, 0.8019757266740498,}},
                                                                                      TestingPair{.in = {3.3, 1.55}, .out = {3.3, 1.55}},
                                                                                      TestingPair{.in = {3.0, -.05, -2}, .out = {3.0, -0.05362956178508537, -2}}

                                                                    }},
                                         (MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::EdgeClusteringMapper>(2, -.1, 1.55, 3.0); },
                                                                    .testingValues = {TestingPair{.in = {0.0, 0.1, .2}, .out = {0.0, 0.1, 0.1828252256954647}}

                                                                    }}),
                         [](const testing::TestParamInfo<MeshMapperTestParameters>& info) { return std::to_string(info.index); });

}  // namespace ablateTesting::domain::modifier