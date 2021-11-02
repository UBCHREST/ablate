#ifndef ABLATELIBRARY_DISTRIBUTEWITHGHOSTCELLS_HPP
#define ABLATELIBRARY_DISTRIBUTEWITHGHOSTCELLS_HPP

#include "modifier.hpp"

namespace ablate::domain::modifier {

class DistributeWithGhostCells : public Modifier {
   private:
    const int ghostCellDepth;

   public:
    explicit DistributeWithGhostCells(int ghostCellDepth = {});

    void Modify(DM&) override;
};

}  // namespace ablate::domain::modifier
#endif  // ABLATELIBRARY_DISTRIBUTEWITHGHOSTCELLS_HPP
