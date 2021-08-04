#include <memory>
#include "gtest/gtest.h"
#include "mathFunctions/functionFactory.hpp"
#include "mathFunctions/linearInterpolator.hpp"

namespace ablateTesting::mathFunctions {

static int ToXFunction(int dim, double time, const double x[], int nf, double* u, void* ctx) {
    u[0] = x[0];
    return 0;
}

TEST(ParsedFunctionTests, ShouldCreateAndParseStream) {
    // arrange
    std::string csvFileString =
        "x, y, z\n"
        ".1, 2.2, 3\n"
        ".2, 2.2, 4\n"
        ".3, 1.1, 2\n";

    std::istringstream csvFileStream(csvFileString);

    // act
    ablate::mathFunctions::LinearInterpolator interpolator(csvFileStream, "x", {"z, y"}, ablate::mathFunctions::Create(ToXFunction));

    // assert
}

}  // namespace ablateTesting::mathFunctions