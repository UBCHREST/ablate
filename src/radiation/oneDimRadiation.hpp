#ifndef ABLATELIBRARY_ONEDIMRADIATION_HPP
#define ABLATELIBRARY_ONEDIMRADIATION_HPP

#include "radiation.hpp"

namespace ablate::radiation {

class OneDimRadiation : public ablate::radiation::Radiation {
   public:
    void Setup(const ablate::solver::Range& cellRange, ablate::domain::SubDomain& subDomain) override;
};
}  // namespace ablate::radiation
#endif  // ABLATELIBRARY_ONEDIMRADIATION_HPP
