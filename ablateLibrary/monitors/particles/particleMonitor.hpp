#ifndef ABLATELIBRARY_PARTICLE_MONITOR_HPP
#define ABLATELIBRARY_PARTICLE_MONITOR_HPP
#include "monitors/monitor.hpp"
#include "particles/particles.hpp"
#include <memory>

namespace ablate::monitors::particles{

class ParticleMonitor : public monitors::Monitor {
   public:
    virtual ~ParticleMonitor() = default;

    virtual void Register(std::shared_ptr<ablate::particles::Particles>) = 0;
};

}

#endif
