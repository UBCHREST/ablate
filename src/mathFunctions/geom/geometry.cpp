#include "geometry.hpp"
#include "mathFunctions/functionFactory.hpp"

ablate::mathFunctions::geom::Geometry::Geometry(const std::shared_ptr<mathFunctions::MathFunction> &insideValuesIn, const std::shared_ptr<mathFunctions::MathFunction> &outsideValuesIn)
    : insideValues(insideValuesIn ? insideValuesIn : ablate::mathFunctions::Create(1.0)),
      outsideValues(outsideValuesIn ? outsideValuesIn : std::make_shared<ablate::mathFunctions::ConstantValue>(0.0)),
      insidePetscFunction(insideValues->GetPetscFunction()),
      insidePetscContext(insideValues->GetContext()),
      outsidePetscFunction(outsideValues->GetPetscFunction()),
      outsidePetscContext(outsideValues->GetContext()) {}

PetscErrorCode ablate::mathFunctions::geom::Geometry::GeometryPetscFunction(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt nf, PetscScalar *u, void *ctx) {
    PetscFunctionBeginUser;
    auto geom = (ablate::mathFunctions::geom::Geometry *)ctx;

    if (geom->InsideGeometry(x, dim, time)) {
        geom->insidePetscFunction(dim, time, x, nf, u, geom->insidePetscContext);
    } else {
        geom->outsidePetscFunction(dim, time, x, nf, u, geom->outsidePetscContext);
    }

    PetscFunctionReturn(0);
}
double ablate::mathFunctions::geom::Geometry::Eval(const double &x, const double &y, const double &z, const double &t) const {
    double temp[3] = {x, y, z};

    return InsideGeometry(temp, 3, t) ? insideValues->Eval(x, y, z, t) : outsideValues->Eval(x, y, z, t);
}

double ablate::mathFunctions::geom::Geometry::Eval(const double *xyz, const int &ndims, const double &t) const {
    return InsideGeometry(xyz, ndims, t) ? insideValues->Eval(xyz, ndims, t) : outsideValues->Eval(xyz, ndims, t);
}

void ablate::mathFunctions::geom::Geometry::Eval(const double &x, const double &y, const double &z, const double &t, std::vector<double> &result) const {
    double temp[3] = {x, y, z};
    if (InsideGeometry(temp, 3, t)) {
        insideValues->Eval(x, y, z, t, result);
    } else {
        outsideValues->Eval(x, y, z, t, result);
    }
}
void ablate::mathFunctions::geom::Geometry::Eval(const double *xyz, const int &ndims, const double &t, std::vector<double> &result) const {
    if (InsideGeometry(xyz, ndims, t)) {
        insideValues->Eval(xyz, ndims, t, result);
    } else {
        outsideValues->Eval(xyz, ndims, t, result);
    }
}

#include "registrar.hpp"
REGISTER_DERIVED(ablate::mathFunctions::MathFunction, ablate::mathFunctions::geom::Geometry);