---
layout: default
title: Testing
parent: Development Guides
nav_order: 11
---

Testing is essential for any high-quality software product and should be integrated at an early stage of development. Primary testing in performed as either unit or integration tests.  Unit testing is designed to test single functions/classes often using mocks.  Unit testing allows to test a much larger set of inputs and expected outputs.  Unit testing lends itself [Test-Driven Development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development).   Integration testing is designed to test entire code functionally where in ABLATE this is usually simulation level inputs.


The [Googletest](https://github.com/google/googletest) framework is used for testing on ABLATE.  The [Googletest Primer](https://google.github.io/googletest/primer.html) is the recommended starting point to become familiar with the framework.  In short, a series of macros `TEST`, `TEST_F`, and `TEST_P` are used to specify standard, text fixtures, or parameterized tests.  The entire test suite can be run with ctest command or with IDEs such as CLion (recommended) or VisualStudio.

There are some useful command line flags that can be used when debugging tests:
- \-\-runMpiTestDirectly=true : when passed in (along with google test single test selection or through CLion run configuration) this flag allows for a test to be run/debug directly.  This bypasses the separate process launch making it easier to debug, but you must directly pass in any needed arguments.
- \-\-keepOutputFile=true : keeps all output files from the tests and reports the file name.

Automated testing is performed on a series of linux Docker images automatically before a pull request can be merged.  If the tests are passing locally but failing for a pull request you can debug using the same environment as the pull request using [Building ABLATE with Docker Directions]({{ site.baseurl }}{%link content/development/BuildingAblateWithDocker.md %}).  Tests can also be run directly in docker using the following commands.

```bash
# Build the docker testing image
docker build -t testing_image -f DockerTestFile .

# or for 64 bit ints
docker build -t testing_image --build-arg PETSC_BUILD_ARCH='arch-ablate-opt-64' -f DockerTestFile .

# Run the built tests and view results
docker run --rm testing_image 

```

## Testing Resources
A series of testing resources and prebuilt fixtures are available to aid in unit and integration testing.  The most commonly of which is the MpiTestFixture and associated classes.  Review the [GoogleTest Advanced Topics](https://google.github.io/googletest/advanced.html#value-parameterized-tests) section for an overview of parametrized tests.  When a textFixture extends the MpiTestFixture fixture, the test will run as a separate mpi process(s).

### Example using a testingResources::MpiTestParamFixture
The ExampleTestFixtureUsingMpi test will run as separate mpi processes for each of the tests described in INSTANTIATE_TEST_SUITE_P. The parallel code must be between `StartWithMPI` and `EndWithMPI`.  Any assertions can be between these macros or after `EndWithMPI`.  Output results can be used for testing by comparing against an expected output file using the `expectedOutputFile` or `expectedFiles` variables in the `MpiTestParameter`.  See [Expected Output Files](#expected-output-files) section for formatting details.
```c++
#include "MpiTestParamFixture.hpp"
#include "gtest/gtest.h"

class ExampleTestFixtureUsingMpi : public testingResources::MpiTestParamFixture {};

TEST_P(ExampleTestFixtureUsingMpi, ShortTestDescription) {
    StartWithMPI
        {
            // arrange
            // initialize petsc and mpi
            PetscInitialize(argc, argv, NULL, NULL) >> testErrorChecker;

            // act
            // Get the current rank
            PetscMPIInt size;
            MPI_Comm_size(PETSC_COMM_WORLD, &size);

            // assert
            ASSERT_EQ(2, size);
        }
        exit(PetscFinalize());
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(ExampleTestFixtureUsingMpi, CsvLogTestFixture,
                         testing::Values((MpiTestParameter){.testName = "mpi test 1", .nproc = 2, .arguments = ""},
                                         (MpiTestParameter){.testName = "mpi test 2", .nproc = 3, .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter> &info) { return info.param.getTestName(); });

```

### Example using a testingResources::MpiTestFixture with custom fixture
For scenarios where additional parameterized information is needed (such as an int, double, or struct) a testingResources::MpiTestFixture can be combined with the GoogleTest ::testing::WithParamInterface as shown.  In this example the example `ExampleTestParameters` are provided to each testing scenario.
```c++
#include "MpiTestParamFixture.hpp"
#include "gtest/gtest.h"

typedef struct {
    testingResources::MpiTestParameter mpiTestParameter;
    double a;
    double b;
    double expectedSum;
} ExampleTestParameters;

class ExampleTestFixture : public testingResources::MpiTestFixture, public ::testing::WithParamInterface<ExampleTestParameters> {
   public:
    void SetUp() override { SetMpiParameters(GetParam().mpiTestParameter); }
};

TEST_P(ExampleTestFixture, ShortDescriptionOfTest) {
    StartWithMPI
        // arrange
        PetscInitialize(argc, argv, NULL, "HELP") >> testErrorChecker;

        const auto& parameters = GetParam();

        // act
        auto sum = parameters.a + parameters.b;

        // assert
        ASSERT_EQ(parameters.expectedSum, sum) << "the sum should match";

        exit(PetscFinalize());
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(ExampleTests, ExampleTestFixture,
                         testing::Values(
                             (ExampleTestParameters){.mpiTestParameter = {.testName = "test 1", .nproc = 1, .arguments = ""},
                                                     .a = 10, .b = 20, .expectedSum = 30},
                             (ExampleTestParameters){.mpiTestParameter = {.testName = "test 2", .nproc = 1, .arguments = ""},
                                                     .a = 20, .b = 20, .expectedSum = 40}
                         ),
                         [](const testing::TestParamInfo<ExampleTestParameters>& info) { return info.param.mpiTestParameter.getTestName(); });

```

## Unit Tests
Unit tests reside in the `tests/ablateLibrary` directory where the test location should match the folder hierarchy in  `ablateLibrary`.  File names should match the corresponding class name followed by `Tests`.  The unit tests should be designed to test single functions/classes where mocks can be used for any required dependency.  Because unit testing should be designed to test a large set of inputs and expected outputs parameterized tests are recommended.

## Integration Tests
Integration tests use the standard yaml input files to test simulation level functions where the output and generated files are compared against expected outputs.  The integration test files also serve as examples for users and are therefor organized by general topic area inside the `inputs` directory.  All integration tests must be rested in the `tests/integrationTests/tests.cpp` file in an existing or new category.  The tests use an `IntegrationTestsSpecifier` struct to specify the required inputs.  The `MpiTestParameter` inside `IntegrationTestsSpecifier` is used to specify:

- .testName: the relative path to the input yaml file starting with `inputs/`
- .nproc: the number of mpi processes to use for testing
- .expectedOutputFile: the expected output file to compare against standard out.  This file also includes a list of created files at the end of the simulation. See [Expected Output Files](#expected-output-files) section for formatting details
- .arguments: optional additional command line arguments to be passed to the simulation
- .expectedFiles: optional list of expected files to compare.  Each entry must be a {"outputs/path/to/file", "nameOfFileInOutputFolder"}

If using the `IntegrationRestartTestsSpecifier` the simulation will be restarted after specify a list of `restartOverrides`.  This is used to test save/restore functionality.

## Expected Output Files
The output log and other files can be compared against expected results for testing.  However, the expected number output may slightly differ based upon compiler/computer configurations so the output format was created.  This file uses your specified [regex](https://www.cplusplus.com/reference/regex/) and compares against an expected value used the specified comparison.  Lines in the expected file that contains `<expects>` are parsed and compared against the provided numbers. Online c++ regex testers might be useful for testing.

```
L_2 Error: \[(.*), (.*), (.*)\]<expects> <1E-13 <1E-13 <1E-13
```
would expect three numbers (all less than 1E-13),

```
L_2 Residual: (.*)<expects> <1E-13
```
one value less than 1E-13,

```
Taylor approximation converging at order (.*)<expects> =2
```
and one value equal to 2.  When using the compare tool, you must escape all regex characters on the line being compared.

| Comparison Character | Description                                                                                                                                |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| <                    | the actual value must be less than the specified expected                                                                                  |
| >                    | the actual value must be greater than the specified expected                                                                               |
| =                    | the actual value muse equal the specified value according to ASSERT_DOUBLE_EQ. A nan can be specified to compare with expected nan output. |
| ~                    | any number is accepted                                                                                                                     |
| *                    | any string is accepted                                                                                                                     |