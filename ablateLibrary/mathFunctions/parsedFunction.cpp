#include "parsedFunction.hpp"
#include <algorithm>
#include <petscsys.h>
#include <exception>

ablate::mathFunctions::ParsedFunction::ParsedFunction(std::string functionString) {
    // define the x,y,z and t variables
    parser.DefineVar("x", &coordinate[0]);
    parser.DefineVar("y", &coordinate[1]);
    parser.DefineVar("z", &coordinate[2]);
    parser.DefineVar("t", &time);

    parser.SetExpr(functionString);

    // Test the function
    parser.Eval();
}
double ablate::mathFunctions::ParsedFunction::Eval(const double& x, const double& y, const double& z, const double& t) const {
    coordinate[0] = x;
    coordinate[1] =y;
    coordinate[2] = z;
    time = t;
    return parser.Eval();
}

double ablate::mathFunctions::ParsedFunction::Eval(const double* xyz, const int& ndims, const double& t) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    for(auto i = 0; i < std::min(ndims, 3); i++){
        coordinate[i] = xyz[i];
    }
    time = t;
    return parser.Eval();
}
void ablate::mathFunctions::ParsedFunction::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    coordinate[0] = x;
    coordinate[1] = y;
    coordinate[2] = z;
    time = t;

    int functionSize = 0;
    auto rawResult = parser.Eval(functionSize);

    if(result.size() < functionSize){
        throw std::invalid_argument("The result vector is not sized to hold the function " + parser.GetExpr());
    }

    // copy over
    for(auto i = 0; i < functionSize; i++){
        result[i] = rawResult[i];
    }
}

void ablate::mathFunctions::ParsedFunction::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    for(auto i = 0; i < std::min(ndims, 3); i++){
        coordinate[i] = xyz[i];
    }
    time = t;

    int functionSize = 0;
    auto rawResult = parser.Eval(functionSize);

    if(result.size() < functionSize){
        throw std::invalid_argument("The result vector is not sized to hold the function " + parser.GetExpr());
    }

    // copy over
    for(auto i = 0; i < functionSize; i++){
        result[i] = rawResult[i];
    }
}

PetscErrorCode ablate::mathFunctions::ParsedFunction::ParsedPetscFunction(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt nf, PetscScalar* u, void* ctx) {
    // wrap in try, so we return petsc error code instead of c++ exception
    PetscFunctionBeginUser;
    try{
        auto parser = (ParsedFunction*)ctx;

        // update the coordinates
        parser->coordinate[0] = 0;
        parser->coordinate[1] = 0;
        parser->coordinate[2] = 0;

        for(auto i = 0; i < std::min(dim, 3); i++){
            parser->coordinate[i] = x[i];
        }
        parser->time = time;

        // Evaluate
        int functionSize = 0;
        auto rawResult = parser->parser.Eval(functionSize);

        if(nf < functionSize){
            throw std::invalid_argument("The result vector is not sized to hold the function " + parser->parser.GetExpr());
        }

        // copy over
        for(auto i = 0; i < functionSize; i++){
            u[i] = rawResult[i];
        }

    } catch (std::exception exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }
    PetscFunctionReturn(0);
}
