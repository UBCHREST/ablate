#ifndef ABLATELIBRARY_PROBEINITIALIZER_HPP
#define ABLATELIBRARY_PROBEINITIALIZER_HPP

#include <filesystem>
#include <memory>
#include "environment/runEnvironment.hpp"
#include "probe.hpp"

namespace ablate::monitors::probes {
class ProbeInitializer {
   public:
    virtual ~ProbeInitializer() = default;

    /**
     * Primary interface to ProbeInitializer.
     * @return
     */
    virtual const std::vector<Probe>& GetProbes() const = 0;

    /**
     * Allow the initializer to specify a different output directory
     * @return
     */
    virtual std::filesystem::path GetDirectory() const { return ablate::environment::RunEnvironment::Get().GetOutputDirectory(); }

    /**
     * Optional function to report rpboe information
     * @return
     */
    virtual void Report(MPI_Comm) const {}
};
}  // namespace ablate::monitors::probes
#endif  // ABLATELIBRARY_PROBEINITIALIZER_HPP
