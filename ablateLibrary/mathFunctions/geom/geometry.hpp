#ifndef ABLATELIBRARY_GEOMETRY_HPP
#define ABLATELIBRARY_GEOMETRY_HPP

#include <mathFunctions/mathFunction.hpp>
#include <memory>

namespace ablate::mathFunctions::geom {

class Geometry : public MathFunction {
   private:
    const std::vector<double> insideValues;
    const std::vector<double> outsideValues;

   protected:
    explicit Geometry(std::vector<double> insideValues, std::vector<double> outsideValues = {});

    virtual bool InsideGeometry(const double* xyz, const int& ndims, const double& time) const = 0;

    static PetscErrorCode GeometryPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

   public:
    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return GeometryPetscFunction; }
};

}  // namespace ablate::mathFunctions::geom
#endif  // ABLATELIBRARY_GEOMETRY_HPP
