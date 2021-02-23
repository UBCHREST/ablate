
#include <muParser.h>
#include <petsc.h>
#include <iostream>
#include <memory>
#include "builder.hpp"
#include "flow/incompressibleFlow.hpp"
#include "incompressibleFlow.h"
#include "mesh/boxMesh.hpp"
#include "parameters/mapParameters.hpp"
#include "parser/listing.h"
#include "parser/registrar.hpp"
#include "parser/yamlParser.hpp"
#include "solve/timeStepper.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

using namespace ablate;

int main(int argc, char **args) {
    try
    {
        double var_a = 1;
        mu::Parser p;
        p.DefineVar("a", &var_a);
        p.DefineVar("b", &var_a);
        p.SetExpr("_pi+min(10,a)");

        for (std::size_t a=0; a<100; ++a)
        {
            var_a = a;  // Change value of variable a
            std::cout << p.Eval() << std::endl;
        }
    }
    catch (mu::Parser::exception_type &e)
    {
        std::cout << e.GetMsg() << std::endl;
    }
//    // initialize petsc and mpi
//    PetscInitialize(&argc, &args, NULL, NULL) >> checkError;
//
//    // check to see if we should print options
//    PetscBool printParserOptions = PETSC_FALSE;
//    PetscOptionsGetBool(NULL, NULL, "-printParser", &printParserOptions, NULL) >> checkError;
//    if(printParserOptions){
//        std::cout << parser::Listing::Get() << std::endl;
//    }
//
//    // check to see if we should print options
//    char  filename[PETSC_MAX_PATH_LEN] = "";
//    PetscBool fileSpecified = PETSC_FALSE;
//    PetscOptionsGetString(NULL, NULL, "-file", filename, PETSC_MAX_PATH_LEN, &fileSpecified) >> checkError;
//    if(!fileSpecified){
//        throw std::invalid_argument("the -file must be specified");
//    }
//
//    std::filesystem::path filePath(filename);
//    if(!std::filesystem::exists(filePath)) {
//        throw std::invalid_argument("unable to locate input file: " + filePath.string() );
//    }
//    {
//        // create the yaml parser
//        std::shared_ptr<parser::Factory> parser = std::make_shared<parser::YamlParser>(filePath);
//
//        // run with the parser
//        Builder::Run(parser);
//    }
//    PetscFinalize() >> checkError;
}