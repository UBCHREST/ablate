#ifndef ABLATELIBRARY_PRINTDOMAINSUMMARY_HPP
#define ABLATELIBRARY_PRINTDOMAINSUMMARY_HPP

#include <petsc.h>
#include <parameters/parameters.hpp>
#include <string>
#include "modifier.hpp"
#include "monitors/logs/log.hpp"

namespace ablate::domain::modifiers {
/**
 * Print a domain summary to standard out
 */
class PrintDomainSummary : public Modifier {
   public:
    /**
     * Allow this monitor to also be used as modifier to save results
     */
    void Modify(DM&) override;

    std::string ToString() const override { return "ablate::domain::modifiers::PrintDomainSummary"; }
};
}  // namespace ablate::domain::modifiers
#endif  // ABLATELIBRARY_PRINTDOMAINSUMMARY_HPP
