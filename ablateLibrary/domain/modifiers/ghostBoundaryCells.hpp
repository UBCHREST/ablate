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

    int Priority() const override { return 5; }
};

}  // namespace ablate::domain::modifier
#endif  // ABLATELIBRARY_GHOSTBOUNDARYCELLS_HPP
