#include "parsedFunction.hpp"
#include <algorithm>
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
