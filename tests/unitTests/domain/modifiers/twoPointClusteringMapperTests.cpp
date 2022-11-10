#include <memory>
#include "domain/modifiers/twoPointClusteringMapper.hpp"
#include "mathFunctions/functionFactory.hpp"
#include "meshMapperTestFixture.hpp"
#include "utilities/vectorUtilities.hpp"

namespace ablateTesting::domain::modifier {

INSTANTIATE_TEST_SUITE_P(
    TwoPointClusteringMapperTests, MeshMapperTestFixture,
    testing::Values((MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::TwoPointClusteringMapper>(0, 0.0, 3.3, 4.0, 1.1, .1); },
                                               .testingValues = {TestingPair{.in = {0.1, 0.2}, .out = {0.306467595064691, .2}},
                                                                 TestingPair{.in = {1.2, 0.9}, .out = {1.1307053735828716, .9}},
                                                                 TestingPair{.in = {0, -.1}, .out = {0.0, -.1}},
                                                                 TestingPair{.in = {1.1, 0.825}, .out = {1.1, 0.825}},
                                                                 TestingPair{.in = {3.3, 1.55}, .out = {3.3, 1.55}},
                                                                 TestingPair{.in = {3.0, -.05, -2}, .out = {2.414207584957076, -.05, -2}}}},
                    (MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::TwoPointClusteringMapper>(1, -.1, 1.55, 3.0, 0.825, .05); },
                                               .testingValues = {TestingPair{.in = {0.1, 0.2}, .out = {.1, 0.45487745290244663}},
                                                                 TestingPair{.in = {0.1, 0.9}, .out = {.1, 0.8596668438585238}},
                                                                 TestingPair{.in = {0, -.1}, .out = {0.0, -.1}},
                                                                 TestingPair{.in = {1.1, 0.825}, .out = {1.1, 0.825}},
                                                                 TestingPair{.in = {3.3, 1.55}, .out = {3.3, 1.55}},
                                                                 TestingPair{.in = {3.0, -.05, -2}, .out = {3.0, 0.03250579527359776, -2}}}},
                    (MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::TwoPointClusteringMapper>(2, -.1, 1.55, 3.0, 0.825, -.05); },
                                               .testingValues = {TestingPair{.in = {0.0, 0.1, .2}, .out = {0.0, 0.1, 0.45487745290244663}}}}),
    [](const testing::TestParamInfo<MeshMapperTestParameters>& info) { return std::to_string(info.index); });

}  // namespace ablateTesting::domain::modifier