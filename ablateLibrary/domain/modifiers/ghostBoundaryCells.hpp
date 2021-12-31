#ifndef ABLATELIBRARY_GHOSTBOUNDARYCELLS_HPP
#define ABLATELIBRARY_GHOSTBOUNDARYCELLS_HPP

#include "modifier.hpp"

namespace ablate::domain::modifiers {

class GhostBoundaryCells : public Modifier {
   private:
    const std::string labelName;

   public:
    explicit GhostBoundaryCells(std::string labelName = {});

    void Modify(DM&) override;

    std::string ToString() const override { return "ablate::domain::modifiers::GhostBoundaryCells"; }
};

}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_GHOSTBOUNDARYCELLS_HPP
