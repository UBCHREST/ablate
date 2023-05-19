#ifndef ABLATELIBRARY_DOMAIN_HDF5INITIALIZER_HPP
#define ABLATELIBRARY_DOMAIN_HDF5INITIALIZER_HPP

#include <filesystem>
#include <memory>
#include <utility>
#include <vector>
#include "field.hpp"
#include "initializer.hpp"
#include "mathFunctions/fieldFunction.hpp"

namespace ablate::domain {
/**
 * Initializes the domain using a previous result stored in an hdf5 file
 */
class Hdf5Initializer : public Initializer {
   private:
    std::filesystem::path hdf5Path;

   public:
    /**
     * Create an empty list
     */
    explicit Hdf5Initializer(std::filesystem::path hdf5Path);

    /**
     * Interface to produce the field functions from fields
     */
    [[nodiscard]] std::vector<std::shared_ptr<mathFunctions::FieldFunction>> GetFieldFunctions(const std::vector<domain::Field>& fields) const override;

   private:
    /**
     * Helper class to determine when to unload the hdf5Mesh
     */
    class Hdf5Mesh {
       public:
        // Pointer to the viewer that can be used to load the mesh or other vectors
        PetscViewer petscViewer = nullptr;

        // Pointer to the base dm
        DM dm = nullptr;

        // count the number of cells in this mesh
        PetscInt numberCells = -1;

        /**
         * Try to load in the dm from the hdf5 file
         * @param hdf5Path
         */
        explicit Hdf5Mesh(const std::filesystem::path& hdf5Path);

        /**
         * provide hook to clean up when not needed
         */
        ~Hdf5Mesh();
    };

    /**
     * Helper function
     */
    class Hdf5MathFunction : public ablate::mathFunctions::MathFunction {
       private:
        // the field used to represent this math function (useful for debugging)
        const std::string field;

        // The dm for this specific field/math function
        DM fieldDm = nullptr;

        // create an empty vec
        Vec fieldVec = nullptr;

        // Hold a reference to the base mesh,viewer
        std::shared_ptr<Hdf5Mesh> baseMesh;

        // The number of components in the field
        PetscInt components = -1;

        // The number of dimensions in the dm
        PetscInt dim = -1;

       public:
        /**
         * Load the hdf5 vector
         * @param baseMesh
         * @param field
         */
        Hdf5MathFunction(std::shared_ptr<Hdf5Mesh> baseMesh, std::string field);

        /**
         * cleanup/destroy the mesh, interpolant, vec
         */
        ~Hdf5MathFunction() override;

       private:
        /**
         * Private method that does the interpolation for the provided point
         * @param dim
         * @param x
         * @param Nf
         * @param u
         * @return
         */
        PetscErrorCode Eval(PetscInt dim, const PetscReal x[], PetscScalar* u) const;

        /**
         * The hdf5 static petsc function for this math fucntion
         * @param dim
         * @param time
         * @param x
         * @param Nf
         * @param u
         * @param ctx
         * @return
         */
        static PetscErrorCode Hdf5PetscFunction(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar* u, void* ctx);

       public:
        /**
         * Return a single double value
         * @param x
         * @param y
         * @param z
         * @param t
         * @return
         */
        [[nodiscard]] double Eval(const double& x, const double& y, const double& z, const double& t) const override;

        /**
         * Return a single double value based upon an xyz array
         * @param xyz
         * @param ndims
         * @param t
         * @return
         */
        [[nodiscard]] double Eval(const double* xyz, const int& ndims, const double& t) const override;

        /**
         * Populate a result array
         * @param x
         * @param y
         * @param z
         * @param t
         * @param result
         */
        void Eval(const double& x, const double& y, const double& z, const double& t, std::vector<double>& result) const override;

        /**
         * Populate a result array based upon an xyz array
         * @param xyz
         * @param ndims
         * @param t
         * @param result
         */
        void Eval(const double* xyz, const int& ndims, const double& t, std::vector<double>& result) const override;

        /**
         * Return a raw petsc style function to evaluate this math function
         * @return
         */
        ablate::mathFunctions::PetscFunction GetPetscFunction() override { return Hdf5PetscFunction; }

        /**
         * Return a context for petsc style functions
         * @return
         */
        void* GetContext() override { return this; }

    };
};

}  // namespace ablate::domain

#endif  // ABLATELIBRARY_DOMAIN_HDF5INITIALIZER_HPP
