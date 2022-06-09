#ifndef ABLATELIBRARY_DISTRIBUTEWITHGHOSTCELLS_HPP
#define ABLATELIBRARY_DISTRIBUTEWITHGHOSTCELLS_HPP

#include "modifier.hpp"

namespace ablate::domain::modifiers {

class DistributeWithGhostCells : public Modifier {
   private:
    const int ghostCellDepth;

    /***
     * Tags the mpi ghost cells. This is a duplicate of the DMPlexCreateVTKLabel_Internal call in PETSc but works without calling DMPlexConstructGhostCells
     * @param dm
     * @param dmNew
     * @return
     */
    PetscErrorCode TagMpiGhostCells(DM dmNew);


   public:
    explicit DistributeWithGhostCells(int ghostCellDepth = {});

    void Modify(DM&) override;

    std::string ToString() const override { return "ablate::domain::modifiers::DistributeWithGhostCells"; }
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_DISTRIBUTEWITHGHOSTCELLS_HPP
