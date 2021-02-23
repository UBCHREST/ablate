#ifndef ABLATELIBRARY_PARSEDFUNCTION_HPP
#define ABLATELIBRARY_PARSEDFUNCTION_HPP
#include <muParser.h>

namespace ablate::mathFunctions{
/**
 * simple wrapper to compute a function from a x,y,z string.
 * see https://beltoforion.de/en/muparser/index.php
 */
class ParsedFunction {
   public:
    mutable double coordinate[3] = {0, 0, 0};
    mutable double time = 0.0;
    mu::Parser parser;

   public:
    ParsedFunction(const ParsedFunction&) = delete;
    void operator=(const ParsedFunction&) = delete;

    explicit ParsedFunction(std::string functionString);

    double Eval(const double& x, const double& y, const double& z, const double &t) const;

    double Eval(const double* xyz, const int& ndims, const double &t) const;

    void Eval(const double& x, const double& y, const double& z, const double &t, std::vector<double>& result) const;

    void Eval(const double* xyz, const int& ndims, const double &t, std::vector<double>& result) const;

};
}

#endif  // ABLATELIBRARY_PARSEDFUNCTION_HPP
