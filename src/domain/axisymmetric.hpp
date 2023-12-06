#ifndef ABLATELIBRARY_AXISYMMETRIC_HPP
#define ABLATELIBRARY_AXISYMMETRIC_HPP

#include <memory>
#include <parameters/parameters.hpp>
#include <vector>
#include "domain.hpp"
#include <array>

namespace ablate::domain {

class Axisymmetric : public Domain {
   private:
    static DM CreateAxisymmetricDM(const std::string& name);
    static void ReplaceDm(DM& originalDm, DM& replaceDm);
   public:
    Axisymmetric(const std::string& name, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::vector<std::shared_ptr<modifiers::Modifier>> modifiers,  std::shared_ptr<parameters::Parameters> options = {});

    ~Axisymmetric() override;


   public:
    /**
     * A simple class that is responsible for determine cell/vertex locations in the mesh
     */
    class MeshGenerator{
       private:
        //! Store the start and end location of the mesh
        const std::array<PetscReal, 3> startLocation;
        const std::array<PetscReal, 3> endLocation;

        //! Store the number of wedges, wedges/pie slices in the circle
        const PetscInt numberWedges;
        //! Store the number of slices, slicing of the cylinder along the z axis
        const PetscInt numberSlices;

        //! Compute the number of cells
        const PetscInt numberCells;

        //! And the number of vertices
        const PetscInt numberVertices;

       public:
        /**
         * generate and precompute a bunch of the required parameters
         * @param startLocation
         * @param endLocation
         * @param numberWedges
         * @param numberSlices
         */
        MeshGenerator(std::array<PetscReal, 3> startLocation, std::array<PetscReal, 3> endLocation, PetscInt numberWedges, PetscInt numberSlices);

        /**
         * Add some getters, so we can virtualize this in the future if needed
         * @return
         */
        const PetscInt& GetNumberCells() const{
            return numberCells;
        }

        /**
         * Add some getters, so we can virtualize this in the future if needed
         * @return
         */
        const PetscInt& GetNumberVertices() const{
            return numberVertices;
        }

    };

};
}  // namespace ablate::domain
#endif  // ABLATECLIENTTEMPLATE_AXISYMMETRIC_HPP
