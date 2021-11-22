#include "parsedNested.hpp"
#include "parsedFunction.hpp"

ablate::mathFunctions::ParsedNested::ParsedNested(std::string functionString, std::map<std::string, std::shared_ptr<MathFunction>> nestedFunctionsIn) : formula(functionString) {
    // define the x,y,z and t variables
    parser.DefineVar("x", &coordinate[0]);
    parser.DefineVar("y", &coordinate[1]);
    parser.DefineVar("z", &coordinate[2]);
    parser.DefineVar("t", &time);

    // for every nestedFunction passed in , use it
    for (auto nestedFunction : nestedFunctionsIn) {
        // store the function
        nestedFunctions.push_back(nestedFunction.second);

        // store the pointer
        nestedValues.push_back(std::make_unique<double>(0.0));

        // register this with the parser
        parser.DefineVar(nestedFunction.first, nestedValues.back().get());
    }

    // define any additional helper functions
    ablate::mathFunctions::ParsedFunction::DefineAdditionalFunctions(parser);
    parser.SetExpr(formula);

    // Test the function
    try {
        parser.Eval();
    } catch (mu::Parser::exception_type& exception) {
        throw ablate::mathFunctions::ParsedFunction::ConvertToException(exception);
    }
}

double ablate::mathFunctions::ParsedNested::Eval(const double& x, const double& y, const double& z, const double& t) const {
    coordinate[0] = x;
    coordinate[1] = y;
    coordinate[2] = z;
    time = t;

    // updated the nested functions
    for (std::size_t i = 0; i < nestedValues.size(); i++) {
        *nestedValues[i] = nestedFunctions[i]->Eval(x, y, z, t);
    }

    return parser.Eval();
}

double ablate::mathFunctions::ParsedNested::Eval(const double* xyz, const int& ndims, const double& t) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    for (auto d = 0; d < std::min(ndims, 3); d++) {
        coordinate[d] = xyz[d];
    }
    time = t;

    // updated the nested functions
    for (std::size_t i = 0; i < nestedValues.size(); i++) {
        *nestedValues[i] = nestedFunctions[i]->Eval(xyz, ndims, t);
    }

    return parser.Eval();
}

void ablate::mathFunctions::ParsedNested::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    coordinate[0] = x;
    coordinate[1] = y;
    coordinate[2] = z;
    time = t;

    // updated the nested functions
    for (std::size_t i = 0; i < nestedValues.size(); i++) {
        *nestedValues[i] = nestedFunctions[i]->Eval(x, y, z, t);
    }

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

void ablate::mathFunctions::ParsedNested::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    // updated the nested functions
    for (std::size_t i = 0; i < nestedValues.size(); i++) {
        *nestedValues[i] = nestedFunctions[i]->Eval(xyz, ndims, t);
    }

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

PetscErrorCode ablate::mathFunctions::ParsedNested::ParsedPetscNested(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt nf, PetscScalar* u, void* ctx) {
    // wrap in try, so we return petsc error code instead of c++ exception
    PetscFunctionBeginUser;
    try {
        auto parser = (ParsedNested*)ctx;

        // update the coordinates
        parser->coordinate[0] = 0;
        parser->coordinate[1] = 0;
        parser->coordinate[2] = 0;

        for (PetscInt i = 0; i < PetscMin(dim, 3); i++) {
            parser->coordinate[i] = x[i];
        }
        parser->time = time;

        // updated the nested functions
        for (std::size_t i = 0; i < parser->nestedValues.size(); i++) {
            parser->nestedFunctions[i]->GetPetscFunction()(dim, time, x, 1, parser->nestedValues[i].get(), parser->nestedFunctions[i]->GetContext());
        }

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

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::ParsedNested,
         " computes string function with variables x, y, z, and t where additional variables can be specified using other functions",
         ARG(std::string, "formula", "see ParsedFunction for details on the string formatting."),
         ARG(std::map<std::string TMP_COMMA ablate::mathFunctions::MathFunction>, "nested", "a map of nested MathFunctions.  These functions are assumed to compute a single scalar value"));