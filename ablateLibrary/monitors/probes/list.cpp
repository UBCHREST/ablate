#include "list.hpp"
#include "utilities/vectorUtilities.hpp"

ablate::monitors::probes::List::List(const std::vector<std::shared_ptr<Probe>>& probes) : List(utilities::VectorUtilities::Copy(probes)) {}
ablate::monitors::probes::List::List(std::vector<Probe> probes) : list(probes) {}

#include "registrar.hpp"
REGISTER_DEFAULT_PASS_THROUGH(ablate::monitors::probes::ProbeInitializer, ablate::monitors::probes::List, "A simple list of probes", std::vector<ablate::monitors::probes::Probe>);
