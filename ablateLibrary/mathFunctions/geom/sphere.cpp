#include "sphere.hpp"
#include <petsc.h>
ablate::mathFunctions::geom::Sphere::Sphere(std::vector<double> center, double radius, const std::shared_ptr<mathFunctions::MathFunction> &insideValues,
                                            const std::shared_ptr<mathFunctions::MathFunction> &outsideValues)
    : Geometry(insideValues, outsideValues),
      center(center),
      radius(radius)

{}

bool ablate::mathFunctions::geom::Sphere::InsideGeometry(const double *xyz, const int &ndims, const double &) const {
    double dist = 0.0;
    for (std::size_t i = 0; i < PetscMin((std::size_t)ndims, center.size()); i++) {
        dist += PetscSqr(xyz[i] - center[i]);
    }
    dist = PetscSqrtReal(dist);

    return dist <= radius;
}

#include "registrar.hpp"
REGISTER(ablate::mathFunctions::MathFunction, ablate::mathFunctions::geom::Sphere, "assigns a uniform value to all points inside the sphere", ARG(std::vector<double>, "center", "the sphere center"),
         OPT(double, "radius", "the sphere radius"), OPT(ablate::mathFunctions::MathFunction, "insideValues", "the values for inside the sphere, defaults to 1"),
         OPT(ablate::mathFunctions::MathFunction, "outsideValues", "the outside values, defaults to zero"));
