#include "parsedSeries.hpp"
#include <petscsys.h>
#include <algorithm>
#include <exception>
#include "parsedFunction.hpp"

ablate::mathFunctions::ParsedSeries::ParsedSeries(std::string functionString, int lowerBound, int upperBound) : formula(functionString), lowerBound(lowerBound), upperBound(upperBound) {
    // define the x,y,z and t variables
    parser.DefineVar("x", &coordinate[0]);
    parser.DefineVar("y", &coordinate[1]);
    parser.DefineVar("z", &coordinate[2]);
    parser.DefineVar("t", &time);
    parser.DefineVar("i", &i);

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

double ablate::mathFunctions::ParsedSeries::Eval(const double& x, const double& y, const double& z, const double& t) const {
    coordinate[0] = x;
    coordinate[1] = y;
    coordinate[2] = z;
    time = t;
    double sum = 0.0;

    for (i = lowerBound; i <= upperBound; i++) {
        sum += parser.Eval();
    }

    return sum;
}

double ablate::mathFunctions::ParsedSeries::Eval(const double* xyz, const int& ndims, const double& t) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    for (auto d = 0; d < std::min(ndims, 3); d++) {
        coordinate[d] = xyz[d];
    }
    time = t;

    double sum = 0.0;

    for (i = lowerBound; i <= upperBound; i++) {
        sum += parser.Eval();
    }

    return sum;
}
void ablate::mathFunctions::ParsedSeries::Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const {
    coordinate[0] = x;
    coordinate[1] = y;
    coordinate[2] = z;
    time = t;

    // zero out the result
    std::fill(result.begin(), result.end(), 0.0);

    // perform multiple evals
    for (i = lowerBound; i <= upperBound; i++) {
        int functionSize = 0;
        auto rawResult = parser.Eval(functionSize);

        if ((int)result.size() < functionSize) {
            throw std::invalid_argument("The result vector is not sized to hold the function " + parser.GetExpr());
        }

        // copy over
        for (auto d = 0; d < functionSize; d++) {
            result[d] += rawResult[d];
        }
    }
}

void ablate::mathFunctions::ParsedSeries::Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const {
    coordinate[0] = 0;
    coordinate[1] = 0;
    coordinate[2] = 0;

    for (auto d = 0; d < std::min(ndims, 3); d++) {
        coordinate[d] = xyz[d];
    }
    time = t;

    // zero out the result
    std::fill(result.begin(), result.end(), 0.0);

    // perform multiple evals
    for (i = lowerBound; i <= upperBound; i++) {
        int functionSize = 0;
        auto rawResult = parser.Eval(functionSize);

        if ((int)result.size() < functionSize) {
            throw std::invalid_argument("The result vector is not sized to hold the function " + parser.GetExpr());
        }

        // copy over
        for (auto d = 0; d < functionSize; d++) {
            result[d] += rawResult[d];
        }
    }
}

PetscErrorCode ablate::mathFunctions::ParsedSeries::ParsedPetscSeries(PetscInt dim, PetscReal time, const PetscReal* x, PetscInt nf, PetscScalar* u, void* ctx) {
    // wrap in try, so we return petsc error code instead of c++ exception
    PetscFunctionBeginUser;
    try {
        auto parser = (ParsedSeries*)ctx;

        // update the coordinates
        parser->coordinate[0] = 0;
        parser->coordinate[1] = 0;
        parser->coordinate[2] = 0;

        for (PetscInt d = 0; d < PetscMin(dim, 3); d++) {
            parser->coordinate[d] = x[d];
        }
        parser->time = time;

        // zero out the u vector
        for (PetscInt f = 0; f < nf; f++) {
            u[f] = 0.0;
        }

        // perform multiple evals
        for (parser->i = parser->lowerBound; parser->i <= parser->upperBound; parser->i++) {
            int functionSize = 0;
            auto rawResult = parser->parser.Eval(functionSize);

            if (nf < functionSize) {
                throw std::invalid_argument("The result vector is not sized to hold the function " + parser->parser.GetExpr());
            }

            // copy over
            for (auto d = 0; d < functionSize; d++) {
                u[d] += rawResult[d];
            }
        }

    } catch (std::exception& exception) {
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, exception.what());
    }
    PetscFunctionReturn(0);
}

#include "parser/registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::ParsedSeries,
         " computes a series result from a string function with variables x, y, z, t, and i where i index of summation. $$\\sum_{i = m}^n formula(x, y, z, t, n)$$",
         ARG(std::string, "formula", "see ParsedFunction for details on the string formatting."), ARG(int, "lowerBound", "the inclusive lower bound of summation (m)"),
         ARG(int, "upperBound", "the inclusive upper bound of summation (n)"));