#include "functionPointer.hpp"
#include <array>
ablate::mathFunctions::FunctionPointer::FunctionPointer(ablate::mathFunctions::PetscFunction function, void *context) : function(function), context(context) {}
double ablate::mathFunctions::FunctionPointer::Eval(const double &x, const double &y, const double &z, const double &t) const {
    std::array<double, 3> loc = {x, y, z};
    double result;
    function(loc.size(), t, &loc[0], 1, &result, context);
    return result;
}
double ablate::mathFunctions::FunctionPointer::Eval(const double *xyz, const int &ndims, const double &t) const {
    double result;
    function(ndims, t, xyz, 1, &result, context);
    return result;
}
void ablate::mathFunctions::FunctionPointer::Eval(const double &x, const double &y, const double &z, const double &t, std::vector<double> &result) const {
    std::array<double, 3> loc = {x, y, z};
    function(3, t, &loc[0], result.size(), &result[0], context);
}
void ablate::mathFunctions::FunctionPointer::Eval(const double *xyz, const int &ndims, const double &t, std::vector<double> &result) const {
    function(ndims, t, xyz, result.size(), &result[0], context);
}
