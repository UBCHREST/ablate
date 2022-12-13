#include <memory>
#include "mathFunctions/functionFactory.hpp"
#include "meshMapperTestFixture.hpp"
#include "utilities/vectorUtilities.hpp"

namespace ablateTesting::domain::modifier {
TEST_P(MeshMapperTestFixture, ShouldModifyValue) {
    // arrange
    auto mapper = GetParam().createMapper();

    // act/assert
    for (const auto& testingValue : GetParam().testingValues) {
        std::vector<double> computed;
        mapper->Modify(testingValue.in, computed);

        // Arrays should be the same
        ASSERT_EQ(testingValue.out.size(), computed.size()) << "The computed values should be the correct length for " << ablate::utilities::VectorUtilities::Concatenate(testingValue.in) << " -> "
                                                            << ablate::utilities::VectorUtilities::Concatenate(testingValue.out) << " for mapper " << mapper->ToString();
        for (std::size_t n = 0; n < testingValue.out.size(); ++n) {
            ASSERT_NEAR(testingValue.out[n], computed[n], 1E-8) << "The computed values for index " << std::to_string(n) << " should be the correct length for mapper " << mapper->ToString();
        }
    }
}

INSTANTIATE_TEST_SUITE_P(MeshMapperTests, MeshMapperTestFixture,
                         testing::Values((MeshMapperTestParameters){.createMapper = []() { return std::make_shared<ablate::domain::modifiers::MeshMapper>(ablate::mathFunctions::Create("x, y")); },
                                                                    .testingValues = {TestingPair{.in = {1.0, 2.0}, .out = {1.0, 2.0}}}},
                                         (MeshMapperTestParameters){
                                             .createMapper = []() { return std::make_shared<ablate::domain::modifiers::MeshMapper>(ablate::mathFunctions::Create("x^2, y^3, z-0.5")); },
                                             .testingValues = {TestingPair{.in = {1.5, 2.5, 3.5}, .out = {2.25, 15.625, 3.}}}}),
                         [](const testing::TestParamInfo<MeshMapperTestParameters>& info) { return std::to_string(info.index); });

}  // namespace ablateTesting::domain::modifier