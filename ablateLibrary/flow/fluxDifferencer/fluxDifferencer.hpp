#ifndef ABLATELIBRARY_FLUXDIFFERENCER_HPP
#define ABLATELIBRARY_FLUXDIFFERENCER_HPP
#include "fluxDifferencer.h"

namespace ablate::flow::fluxDifferencer {
class FluxDifferencer {
   public:
    virtual ~FluxDifferencer() = default;
    virtual FluxDifferencerFunction GetFluxDifferencerFunction() = 0;
};
}  // namespace ablate::flow::fluxDifferencer
#endif  // ABLATELIBRARY_FLUXDIFFERENCER_HPP
