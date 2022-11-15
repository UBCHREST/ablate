#include "equationInterval.hpp"

ablate::io::interval::EquationInterval::EquationInterval(std::string functionString): ablate::mathFunctions::FormulaBase(functionString, {}) {
    // Remove the x, y, z for this implmentation
    parser.ClearVar();

    // add back the two required vars
    parser.DefineVar("step", &step);
    parser.DefineVar("time", &time);

    // set the expression
    parser.SetExpr(functionString);
}
bool ablate::io::interval::EquationInterval::Check(MPI_Comm comm, PetscInt stepIn, PetscReal timeIn) {
    // updated the linked variables
    step = stepIn;
    time = timeIn;

    return parser.Eval() > 0;
}

#include "registrar.hpp"
REGISTER_PASS_THROUGH(ablate::io::interval::Interval, ablate::io::interval::EquationInterval, "Determines if the interval is valid based upon a user supplied equation. If the result of the equation is > 0, then check will be true, else it will be false.  The custom variables for this formula are \"time\" and \"step\"", std::string);

