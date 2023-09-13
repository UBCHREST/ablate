---
layout: default
title: Testing
parent: Code Development
nav_order: 3
---

Testing is essential for any high-quality software product and should be integrated at an early stage of development. Primary testing in performed as either unit or integration tests.  Unit testing is designed to test single functions/classes often using mocks.  Unit testing allows to test a much larger set of inputs and expected outputs.  Unit testing lends itself [Test-Driven Development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development).   Integration testing is designed to test entire code functionally where in ABLATE this is usually simulation level inputs.


The [Googletest](https://github.com/google/googletest) framework is used for testing on ABLATE.  The [Googletest Primer](https://google.github.io/googletest/primer.html) is the recommended starting point to become familiar with the framework.  In short, a series of macros `TEST`, `TEST_F`, and `TEST_P` are used to specify standard, text fixtures, or parameterized tests.  The entire test suite can be run with ctest command or with IDEs such as CLion (recommended) or VisualStudio.

There are some useful command line flags that can be used when debugging tests:
- \-\-runMpiTestDirectly=true : when passed in (along with google test single test selection or through CLion run configuration) this flag allows for a test to be run/debug directly.  This bypasses the separate process launch making it easier to debug, but you must directly pass in any needed arguments.
- \-\-keepOutputFile=true : keeps all output files from the tests and reports the file name.

Automated testing is performed on a series of linux Docker images automatically before a pull request can be merged.  If the tests are passing locally but failing for a pull request you can debug using the same environment as the pull request using [Docker Install]({{ site.baseurl }}{%link content/installation/DockerInstall.md %}).  Tests can also be run directly in docker using the following commands.

```bash
# Build the default docker testing image
docker build -t testing_image -f DockerTestFile .

# or select a specific base version from https://github.com/orgs/UBCHREST/packages?tab=packages&q=ablate-dependencies
docker build -t testing_image --build-arg ABLATE_DEPENDENCY_IMAGE=ghcr.io/ubchrest/ablate/ablate-dependencies-clang-index64:latest -f DockerTestFile .

# Run the built tests and view results
docker run --rm testing_image 

# Run the built tests including regression tests
docker run --rm testing_image ctest
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
            Initialize(argc, argv, NULL, NULL) >> testErrorChecker;

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
        Initialize(argc, argv, NULL, "HELP") >> testErrorChecker;

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
Unit tests reside in the `tests/unitTests` directory where the test location should match the folder hierarchy in `src`.  File names should match the corresponding class name followed by `Tests`.  The unit tests should be designed to test single functions/classes where mocks can be used for any required dependency.  Because unit testing should be designed to test a large set of inputs and expected outputs parameterized tests are recommended.

## Integration Tests
Integration tests use the standard yaml input files to test simulation level functions where the output and generated files are compared against expected outputs.  The integration test files also serve as examples for users and are therefor organized by general topic area inside the `inputs` directory. Any comments before the first '---' will be treated ad markdown (with the # removed) when the documentation is generated and should be written as such.  Math and equations are rendered using [MathJax](https://www.mathjax.org) using Latex style equations where $$ is used to define math regions.  All input files in this directory will be used for integration tests and must specify the testing parameters.  The default test is a standard integration test shown:

```yaml
# metadata used only for integration testing
test:
  # a unique test name for this integration tests
  name: aUniqueNameForThisTest
  # optionally specify the number of mpi ranks
  ranks: 2
  # optionally provide overrides for asan testing (this is not common)
  environment: ""
  # optional additional arguments
  arguments: ""
  
  # specify a single
  assert: "inputs/compressibleFlow/compressibleCouetteFlow.txt"
  
  # or list of asserts to compare to the output
  asserts:
    - "inputs/compressibleFlow/compressibleCouetteFlow.txt"
    - !testingResources::asserts::TextFileAssert
      expected: "inputs/reactingFlow/ignitionDelayGriMech.Temperature.txt"
      actual: "ignitionDelayTemperature.csv"
```

If the same input file is used for more than a single test, the ```tests``` keyword can be used instead.
```yaml
# metadata used only for integration testing
tests:
  - # a unique test name for this integration tests
    name: aUniqueNameForThisTest1
    # specify a single assert
    assert: "inputs/compressibleFlow/compressibleCouetteFlow.txt"
  - # a unique test name for this integration tests
    name: aUniqueNameForThisTest2
    # optional additional arguments
    arguments: "--change-input=true"
    # specify a single assert
    assert: "inputs/compressibleFlow/compressibleCouetteFlowOutput2.txt"
```

An advanced restart integration test can also be specified instead of the default.  This is used when restarting the simulation is part of the test.  All the same asserts can be used.  Integration restart and standard tests can be used in the same list.

```yaml
test: !IntegrationRestartTest
    # specify the basic test parameters for restart test
    testParameters:
      # a unique test name for this integration tests
      name: incompressibleFlowIntervalRestart
      asserts:
        - "inputs/feFlow/incompressibleFlowIntervalRestart.txt"
    # optionally upon restart, override some of the input parameters
    restartOverrides:
      timestepper::arguments::ts_max_steps: "10"
    
    # optionally upon restart, use a separate input file
    restartInputFile: "inputs/compressibleFlow/hdf5InitializerTest/hdf5InitializerTest.Initialization.yaml"
```

Any number of asserts can be listed and combined to complete the integration test.  Available asserts include:

### testingResources::asserts::StdOutAssert (Default)
This is the default assert so the name does not need to be specified.  It compares the output of standard out to an expected file. However, the expected number output may slightly differ based upon compiler/computer configurations so the output format was created.  This file uses your specified [regex](https://www.cplusplus.com/reference/regex/) and compares against an expected value used the specified comparison.  Lines in the expected file that contains `<expects>` are parsed and compared against the provided numbers. Online c++ regex testers might be useful for testing. 

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
| n                    | should be near value (percent difference <=1E-3)                                                                                           |
| z                    | should be near zero (<1E-13)                                                                                                               |

The only required input for this file is the relative path to the expected test file. 

```yaml
      # Example as a single assert
      assert: "inputs/feFlow/incompressibleFlowIntervalRestart.txt"

      # Example as part of a list of asserts
      asserts:
        - "inputs/feFlow/incompressibleFlowIntervalRestart.txt"
```

### testingResources::asserts::TextFileAssert
The text file assert is used compare text files generated by ABLATE to expected files.  The text comparisons are handled the same as [testingResources::asserts::StdOutAssert](#testingResources::asserts::StdOutAssert).  The required inputs are path to the expected file and generated/actual file name/path.  The generated/actual path to relative to the output directory.
```yaml
      # as a list of asserts
      asserts:
        - !testingResources::asserts::TextFileAssert
          expected: "inputs/feFlow/incompressibleFlowRestartProbe.csv"
          actual: "incompressibleFlowRestartProbe.csv"
      # as a single assert
      assert: !testingResources::asserts::TextFileAssert
        expected: "inputs/feFlow/incompressibleFlowRestartProbe.csv"
        actual: "incompressibleFlowRestartProbe.csv"
```

### testingResources::asserts::SolutionVectorAssert
The SolutionVectorAssert reads in the solution vector from the specified (actual) file and compared against a saved hdf5 solution.

```yaml
# specify an assert to compare the expected solution vector to a hdf5 saved vector
assert: !testingResources::asserts::SolutionVectorAssert
    expected: "inputs/levelSet/2D_Ellipse.hdf5" # relative path to the expected input file
    actual: "domain.hdf5" # the actual file in the output directory
    type: linf # the norm type for comparisons. Options include ('l1','l1_norm','l2', 'linf', 'l2_norm')
    tolerance: 1E-8 # optional tolerance, default is 1E-12
```


## Regression Tests
Regression tests operate similarly to the Integration Tests but are not run as part of the pull request process.  Instead, they run on an automated schedule.  These larger/longer simulations are used to ensure that ABLATE functionally does not regress and serve as well documented examples of using ABLATE with real world problems.  They are setup and controlled the same as Integration Tests.
