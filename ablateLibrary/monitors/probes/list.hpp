#ifndef ABLATELIBRARY_PROBEINITIALIZER_LIST_HPP
#define ABLATELIBRARY_PROBEINITIALIZER_LIST_HPP

#include <memory>
#include <vector>
#include "probeInitializer.hpp"

namespace ablate::monitors::probes {

class List : public ProbeInitializer {
   private:
    const std::vector<Probe> list;

   public:
    /**
     * default list of probe monitors, useful for the parser init
     * @param probes
     */
    explicit List(const std::vector<std::shared_ptr<Probe>>& probes);

    /**
     * list of probe monitors
     * @param probes
     */
    explicit List(std::vector<Probe> probes);

    /**
     * list of probe locations with names
     * @return
     */
    const std::vector<Probe>& GetProbes() const override { return list; }
};

}  // namespace ablate::monitors::probes
#endif  // ABLATELIBRARY_LIST_HPP
