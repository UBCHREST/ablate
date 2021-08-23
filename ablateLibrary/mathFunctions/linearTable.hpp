#ifndef ABLATELIBRARY_LINEARTABLE_HPP
#define ABLATELIBRARY_LINEARTABLE_HPP

#include <filesystem>
#include <istream>
#include <vector>
#include "mathFunction.hpp"
namespace ablate::mathFunctions {

/**
 * a simple table that reads a text file and interpolates the value.  The xAxisColumn is assumed to be monotonic.
 * An example input would look like
 * x,y,z
 * 1,2,3
 * 2,2,1
 */
class LinearTable : public MathFunction {
   private:
    std::vector<double> independentValues;
    std::vector<std::vector<double>> dependentValues;
    const std::string independentColumnName;
    const std::vector<std::string> dependentColumnsNames;
    const std::shared_ptr<MathFunction> independentValueFunction;

   private:
    void ParseInputData(std::istream& inputFile);

    void Interpolate(double x, size_t numInterpolations, double* result) const;

    static PetscErrorCode LinearInterpolatorPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    LinearTable(std::filesystem::path inputFile, std::string independentColumnName, std::vector<std::string> dependentColumnsNames, std::shared_ptr<MathFunction> independentValueFunction);
    LinearTable(std::istream& inputStream, std::string independentColumnName, std::vector<std::string> dependentColumnsNames, std::shared_ptr<MathFunction> independentValueFunction);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return LinearInterpolatorPetscFunction; }

    const std::vector<double>& GetIndependentValues() const { return independentValues; }

    const std::vector<std::vector<double>>& GetDependentValues() const { return dependentValues; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_LINEARTABLE_HPP
