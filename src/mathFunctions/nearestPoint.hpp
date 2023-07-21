#ifndef ABLATELIBRARY_NEARESTPOINT_HPP
#define ABLATELIBRARY_NEARESTPOINT_HPP

#include <filesystem>
#include <istream>
#include <vector>
#include "mathFunction.hpp"
namespace ablate::mathFunctions {

/**
 * Simple math function that takes a list of a points/values and returns the value associated with the nearest point
 */
class NearestPoint : public MathFunction {
   private:
    //! list of coordinates (x1, y1, z1, x2, y2, etc.)
    const std::vector<double> coordinates;

    //! List of values
    const std::vector<double> values;

    //! The dimension of the coordinates
    const std::size_t dimension;

    //! The dimension of the coordinates
    const std::size_t numberPoints;

   private:
    /**
     * static call to be called from petsc
     * @param dim
     * @param time
     * @param x
     * @param Nf
     * @param u
     * @param ctx
     * @return
     */
    static PetscErrorCode NearestPointPetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

    /**
     * Call to find the nearest point in xyz
     * @param xyz
     * @param xyzDimension
     * @return
     */
    std::size_t FindNearestPoint(const double* xyz, std::size_t xyzDimension) const;

   public:
    /**
     * Create a simple math function that initializes based upon the list of points
     * @param coordinates list of coordinates (x1, y1, z1, x2, y2, etc.)
     * @param values List of values
     */
    NearestPoint(const std::vector<double>& coordinates, const std::vector<double>& values);

    double Eval(const double& x, const double& y, const double& z, const double& t) const override;

    double Eval(const double* xyz, const int& ndims, const double& t) const override;

    void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

    void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

    void* GetContext() override { return this; }

    PetscFunction GetPetscFunction() override { return NearestPointPetscFunction; }
};
}  // namespace ablate::mathFunctions
#endif  // ABLATELIBRARY_NEARESTPOINT_HPP
