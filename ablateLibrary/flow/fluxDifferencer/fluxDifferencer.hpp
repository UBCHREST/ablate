#ifndef ABLATELIBRARY_FLUXDIFFERENCER_HPP
#define ABLATELIBRARY_FLUXDIFFERENCER_HPP
#include "fluxDifferencer.h"

namespace ablate::flow::fluxDifferencer {
class FluxDifferencer {
   public:
    virtual FluxDifferencerFunction GetFluxDifferencerFunction() = 0;
};
}
#endif  // ABLATELIBRARY_FLUXDIFFERENCER_HPP
