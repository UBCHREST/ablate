#include "simpleFormula.hpp"
#include <petscsys.h>
#include <algorithm>
#include <exception>

static mu::value_type powerFunction(mu::value_type a, mu::value_type b) { return PetscPowReal(a, b); }

ablate::mathFunctions::SimpleFormula::SimpleFormula(std::string functionString) : formula(functionString) {
    // define the x,y,z and t variables
    parser.DefineVar("x", &coordinate[0]);
    parser.DefineVar("y", &coordinate[1]);
    parser.DefineVar("z", &coordinate[2]);
    parser.DefineVar("t", &time);

    // define any additional helper functions
    DefineAdditionalFunctions(parser);

    parser.SetExpr(formula);

    // Test the function
    try {
        parser.Eval();
    } catch (mu::Parser::exception_type& exception) {
        throw ablate::mathFunctions::SimpleFormula::ConvertToException(exception);
    }
}
double ablate::mathFunctions::SimpleFormula::Eval(const double& x, const double& y, const double& z, const double& t) const {
    coordinate[0] = x;
    coordinate[1] = y;
    coordinate[2] = z;
    time = t;
    return parser.Eval();
}

double ablate::mathFunctions::SimpleFormula::Eval(const double* xyz, const int& ndims, const double& t) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    for (auto i = 0; i < std::min(ndims, 3); i++) {
        coordinate[i] = xyz[i];
    }
    time = t;
    return parser.Eval();
}

void ablate::mathFunctions::SimpleFormula::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    coordinate[0] = x;
    coordinate[1] = y;
    coordinate[2] = z;
    time = t;

    int functionSize = 0;
    auto rawResult = parser.Eval(functionSize);

    if ((int)result.size() < functionSize) {
        throw std::invalid_argument("The result vector is not sized to hold the function " + parser.GetExpr());
    }

    // copy over
    for (auto i = 0; i < functionSize; i++) {
        result[i] = rawResult[i];
    }
}

void ablate::mathFunctions::SimpleFormula::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    for (auto i = 0; i < std::min(ndims, 3); i++) {
        coordinate[i] = xyz[i];
    }
    time = t;

    int functionSize = 0;
    auto rawResult = parser.Eval(functionSize);

    if ((int)result.size() < functionSize) {
        throw std::invalid_argument("The result vector is not sized to hold the function " + parser.GetExpr());
    }

    // copy over
    for (auto i = 0; i < functionSize; i++) {
        result[i] = rawResult[i];
    }
}

PetscErrorCode ablate::mathFunctions::SimpleFormula::ParsedPetscFunction(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt nf, PetscScalar* u, void* ctx) {
    // wrap in try, so we return petsc error code instead of c++ exception
    PetscFunctionBeginUser;
    try {
        auto parser = (SimpleFormula*)ctx;

        // update the coordinates
        parser->coordinate[0] = 0;
        parser->coordinate[1] = 0;
        parser->coordinate[2] = 0;

        for (PetscInt i = 0; i < PetscMin(dim, (PetscInt)3); i++) {
            parser->coordinate[i] = x[i];
        }
        parser->time = time;

        // Evaluate
        int functionSize = 0;
        auto rawResult = parser->parser.Eval(functionSize);

        if (nf < functionSize) {
            throw std::invalid_argument("The result vector is not sized to hold the function " + parser->parser.GetExpr());
        }

        // copy over
        for (auto i = 0; i < functionSize; i++) {
            u[i] = rawResult[i];
        }

    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }
    PetscFunctionReturn(0);
}

void ablate::mathFunctions::SimpleFormula::DefineAdditionalFunctions(mu::Parser& parser) {
    // define some helper functions
    parser.DefineFun("Power", powerFunction, true);
}

std::invalid_argument ablate::mathFunctions::SimpleFormula::ConvertToException(mu::Parser::exception_type& exception) {
    return std::invalid_argument("Unable to parser (" + exception.GetExpr() + "). " + exception.GetMsg());
}

#include "registrar.hpp"
REGISTER_DEFAULT_PASS_THROUGH(ablate::mathFunctions::MathFunction, ablate::mathFunctions::SimpleFormula,
                              "a string based function to be parsed with muparser. The (string) formula that may accept x, y, z, t as variables", std::string);