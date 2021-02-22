
#include <petsc.h>
#include <memory>
#include "flow/incompressibleFlow.hpp"
#include "incompressibleFlow.h"
#include "mesh/boxMesh.hpp"
#include "parameters/mapParameters.hpp"
#include "parser/yamlParser.hpp"
#include "parser/listing.h"
#include "parser/registrar.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"
#include "builder.hpp"

using namespace ablate;

int main(int argc, char **args) {
    // initialize petsc and mpi
    PetscInitialize(&argc, &args, NULL, NULL) >> checkError;

    // check to see if we should print options
    PetscBool printParserOptions = PETSC_FALSE;
    PetscOptionsGetBool(NULL, NULL, "-printParser", &printParserOptions, NULL) >> checkError;
    if(printParserOptions){
        std::cout << parser::Listing::Get() << std::endl;
    }

    // check to see if we should print options
    char  filename[PETSC_MAX_PATH_LEN] = "";
    PetscBool fileSpecified = PETSC_FALSE;
    PetscOptionsGetString(NULL, NULL, "-file", filename, PETSC_MAX_PATH_LEN, &fileSpecified) >> checkError;
    if(!fileSpecified){
        throw std::invalid_argument("the -file must be specified");
    }

    std::filesystem::path filePath(filename);
    if(!std::filesystem::exists(filePath)) {
        throw std::invalid_argument("unable to locate input file: " + filePath.string() );
    }
    {
        // create the yaml parser
        std::shared_ptr<parser::Factory> parser = std::make_shared<parser::YamlParser>(filePath);

        // run with the parser
        Builder::Run(parser);
    }
    PetscFinalize() >> checkError;
}