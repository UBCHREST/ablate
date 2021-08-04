#ifndef ABLATELIBRARY_LINEARINTERPOLATOR_HPP
#define ABLATELIBRARY_LINEARINTERPOLATOR_HPP

#include "mathFunction.hpp"
#include <filesystem>
#include <istream>
namespace ablate::mathFunctions {

/**
 * a simple interpolator that reads a text file and interpolates the value.  The xAxisColumn is assumed to be monotonic.
 * An example input would look like
 * x,y,z
 * 1,2,3
 * 2,2,1
 */
class LinearInterpolator : public MathFunction {
   private:
    std::vector<double> xValues;
    std::vector<std::vector<double>> yValues;
    const std::string xColumn;
    const std::vector<std::string> yColumns;
    const std::shared_ptr<MathFunction> locationToXCoordFunction;

   private:
    void ParseInputData(std::istream& inputFile);

   public:
    LinearInterpolator(std::filesystem::path inputFile, std::string xColumn, std::vector<std::string> yColumns, std::shared_ptr<MathFunction> locationToXCoord);
    LinearInterpolator(std::istream& inputFile, std::string xColumn, std::vector<std::string> yColumns, std::shared_ptr<MathFunction> locationToXCoord);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override{return NAN;}

    double Eval(const double* xyz, const int& ndims, const double& t) const override{return NAN;}

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override{}

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override{}

    void* GetContext() override{ return nullptr;}

    PetscFunction GetPetscFunction() override{ return nullptr;}

};
}
#endif  // ABLATELIBRARY_LINEARINTERPOLATOR_HPP
