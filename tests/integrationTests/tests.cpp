static char help[] = "Integration Level Testing";

#include <petsc.h>
#include <filesystem>
#include "MpiTestFixture.hpp"
#include "MpiTestParamFixture.hpp"
#include "builder.hpp"
#include "gtest/gtest.h"
#include "parser/yamlParser.hpp"

/**
 * Note: the test name is assumed to be the relative path to the yaml file
 */
class IntegrationTestsSpecifier : public testingResources::MpiTestParamFixture {};

TEST_P(IntegrationTestsSpecifier, ShouldRun) {
    StartWithMPI
        // initialize petsc and mpi
        PetscErrorCode ierr = PetscInitialize(argc, argv, NULL, help);
        CHKERRABORT(PETSC_COMM_WORLD, ierr);
        {
            // get the file
            std::filesystem::path inputPath = GetParam().testName;

            // load a yaml file
            std::shared_ptr<ablate::parser::Factory> parser = std::make_shared<ablate::parser::YamlParser>(inputPath);

            // run with the parser
            ablate::Builder::Run(parser);
        }
        ierr = PetscFinalize();
        exit(ierr);
    EndWithMPI
}

INSTANTIATE_TEST_SUITE_P(Tests, IntegrationTestsSpecifier,
                         testing::Values((MpiTestParameter){.testName = "inputs/incompressibleFlow.yaml", .nproc = 1, .expectedOutputFile = "outputs/incompressibleFlow.out", .arguments = ""}),
                         [](const testing::TestParamInfo<MpiTestParameter> &info) { return info.param.getTestName(); });
