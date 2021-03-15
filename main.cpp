#include <iostream>
#include <memory>
#include "builder.hpp"
#include "parser/listing.h"
#include "parser/yamlParser.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"
#include "monitors/runEnvironment.hpp"

using namespace ablate;

int main(int argc, char **args) {
    // initialize petsc and mpi
    PetscInitialize(&argc, &args, NULL, NULL) >> checkError;

    // check to see if we should print version
    PetscBool printVersion = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-version", &printVersion, NULL) >> checkError;
    if (printVersion) {
        Builder::PrintVersion(std::cout);
        std::cout << "----------------------------------------" << std::endl;
    }

    // check to see if we should print options
    PetscBool printParserOptions = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "--parserHelp", &printParserOptions, NULL) >> checkError;
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

    std::filesystem::path filePath(filename);
    if (!std::filesystem::exists(filePath)) {
        throw std::invalid_argument("unable to locate input file: " + filePath.string());
    }
    {
        // create the yaml parser
        std::shared_ptr<parser::YamlParser> parser = std::make_shared<parser::YamlParser>(filePath);

        // setup the monitor
        auto setupEnvironmentParameters = parser->GetByName<ablate::parameters::Parameters>("environment");
        monitors::RunEnvironment::Setup(filePath, *setupEnvironmentParameters);

        // run with the parser
        Builder::Run(parser);

        // check for unused parameters
        auto unusedValues = parser->GetUnusedValues();
        if(!unusedValues.empty()){
            std::cout << "WARNING: The following input parameters were not used:" << std::endl;
            for(auto unusedValue : unusedValues){
                std::cout << unusedValue << std::endl;
            }
        }
    }
    PetscFinalize() >> checkError;
}