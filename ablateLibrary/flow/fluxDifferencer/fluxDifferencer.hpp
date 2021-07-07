#ifndef ABLATELIBRARY_FLUXDIFFERENCER_HPP
#define ABLATELIBRARY_FLUXDIFFERENCER_HPP
#include "fluxDifferencer.h"

namespace ablate::flow::fluxDifferencer {
class FluxDifferencer {
   public:
    FluxDifferencer() = default;
    FluxDifferencer(FluxDifferencer const&) = delete;
    FluxDifferencer& operator=(FluxDifferencer const&) = delete;
    virtual ~FluxDifferencer() = default;
    virtual FluxDifferencerFunction GetFluxDifferencerFunction() = 0;
};
}  // namespace ablate::flow::fluxDifferencer
#endif  // ABLATELIBRARY_FLUXDIFFERENCER_HPP
