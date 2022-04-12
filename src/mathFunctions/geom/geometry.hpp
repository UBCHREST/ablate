#ifndef ABLATELIBRARY_GEOMETRY_HPP
#define ABLATELIBRARY_GEOMETRY_HPP

#include <mathFunctions/mathFunction.hpp>
#include <memory>

namespace ablate::mathFunctions::geom {

class Geometry : public MathFunction {
   private:
    const std::shared_ptr<mathFunctions::MathFunction> insideValues;
    const std::shared_ptr<mathFunctions::MathFunction> outsideValues;

    //! Store a reference to the inside petsc function
    const PetscFunction insidePetscFunction;
    void* insidePetscContext;

    //! Store a reference to the outside petsc function
    const PetscFunction outsidePetscFunction;
    void* outsidePetscContext;

   protected:
    explicit Geometry(const std::shared_ptr<mathFunctions::MathFunction>& insideValues, const std::shared_ptr<mathFunctions::MathFunction>& outsideValues);

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
