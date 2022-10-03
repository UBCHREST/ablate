#include "oneDimRadiation.hpp"

void ablate::radiation::OneDimRadiation::Setup(const solver::Range& cellRange, ablate::domain::SubDomain& subDomain) {
    nTheta = 1;  //!< Reduce the number of rays if one dimensional symmetry can be taken advantage of
    ablate::radiation::Radiation::Setup(cellRange, subDomain);
}

