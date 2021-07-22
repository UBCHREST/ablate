#include <fstream>
#include <iostream>
#include <memory>
#include <parameters/petscPrefixOptions.hpp>
#include <utilities/fileUtility.hpp>
#include <utilities/mpiError.hpp>
#include "builder.hpp"
#include "environment/runEnvironment.hpp"
#include "parser/listing.h"
#include "parser/yamlParser.hpp"
#include "utilities/petscError.hpp"

using namespace ablate;

const char* replacementInputPrefix = "-yaml::";

int main(int argc, char** args) {
    // initialize petsc and mpi
    PetscInitialize(&argc, &args, NULL, NULL) >> checkError;

    // check to see if we should print version
    PetscBool printVersion = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-version", &printVersion, NULL) >> checkError;
    if (!printVersion) {
        PetscOptionsGetBool(NULL, NULL, "--version", &printVersion, NULL) >> checkError;
    }
    if (printVersion) {
        Builder::PrintVersion(std::cout);
        std::cout << "----------------------------------------" << std::endl;
    }

    // check to see if we should print options
    PetscBool printParserOptions = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "--help", &printParserOptions, NULL) >> checkError;
    if (printParserOptions) {
        std::cout << parser::Listing::Get() << std::endl;
    }

    // check to see if we should print options
    char filename[PETSC_MAX_PATH_LEN] = "";
    PetscBool fileSpecified = PETSC_FALSE;
    PetscOptionsGetString(NULL, NULL, "--input", filename, PETSC_MAX_PATH_LEN, &fileSpecified) >> checkError;
    if (!fileSpecified) {
        throw std::invalid_argument("the --input must be specified");
    }

    // locate or download the file
    auto filePath = ablate::utilities::FileUtility::LocateFile(filename, PETSC_COMM_WORLD);
    if (!std::filesystem::exists(filePath)) {
        throw std::invalid_argument("unable to locate input file: " + filePath.string());
    }
    {
        // build options from the command line
        auto yamlOptions = std::make_shared<ablate::parameters::PetscPrefixOptions>(replacementInputPrefix);

        // create the yaml parser
        std::shared_ptr<parser::YamlParser> parser = std::make_shared<parser::YamlParser>(filePath, true, yamlOptions);

        // setup the monitor
        auto setupEnvironmentParameters = parser->GetByName<ablate::parameters::Parameters>("environment");
        environment::RunEnvironment::Setup(*setupEnvironmentParameters, filePath);

        // Copy over the input file
        int rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank) >> checkMpiError;
        if (rank == 0) {
            std::filesystem::path inputCopy = environment::RunEnvironment::Get().GetOutputDirectory() / filePath.filename();
            std::ofstream stream(inputCopy);
            parser->Print(stream);
            stream.close();
        }

        // run with the parser
        Builder::Run(parser);

        // check for unused parameters
        auto unusedValues = parser->GetUnusedValues();
        if (!unusedValues.empty()) {
            std::cout << "WARNING: The following input parameters were not used:" << std::endl;
            for (auto unusedValue : unusedValues) {
                std::cout << unusedValue << std::endl;
            }
        }
    }
    PetscFinalize() >> checkError;
}