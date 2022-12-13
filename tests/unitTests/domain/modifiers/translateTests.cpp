#include <memory>
#include "domain/modifiers/translate.hpp"
#include "meshMapperTestFixture.hpp"

namespace ablateTesting::domain::modifier {

// These tests use the ShouldModifyValue function defined in meshMapperTests
INSTANTIATE_TEST_SUITE_P(TranslateTests, MeshMapperTestFixture,
                         testing::Values((MeshMapperTestParameters){.createMapper =
                                                                        []() {
                                                                            return std::make_shared<ablate::domain::modifiers::Translate>(std::vector<double>{.4, .2});
                                                                        },
                                                                    .testingValues = {TestingPair{.in = {1.0, 2.0}, .out = {1.4, 2.2}},
                                                                                      TestingPair{.in = {1.0, 2.0, 4.0}, .out = {1.4, 2.2, 4.0}},
                                                                                      TestingPair{.in = {-.2}, .out = {.2}}}},
                                         (MeshMapperTestParameters){.createMapper =
                                                                        []() {
                                                                            return std::make_shared<ablate::domain::modifiers::Translate>(std::vector<double>{-1.5, 2.0, -1});
                                                                        },
                                                                    .testingValues = {TestingPair{.in = {1.0, 2.0}, .out = {-.5, 4.0}},
                                                                                      TestingPair{.in = {1.0, 2.0, 4.0}, .out = {-.5, 4.0, 3.0}},
                                                                                      TestingPair{.in = {-.2}, .out = {-1.7}}}}),
                         [](const testing::TestParamInfo<MeshMapperTestParameters>& info) { return std::to_string(info.index); });

}  // namespace ablateTesting::domain::modifier