#ifndef ABLATELIBRARY_FLOW_MONITOR_HPP
#define ABLATELIBRARY_FLOW_MONITOR_HPP
#include "monitors/monitor.hpp"
#include "flow/flow.hpp"
#include <memory>

namespace ablate::monitors::flow{

class FlowMonitor : public monitors::Monitor {
   public:
    virtual ~FlowMonitor() = default;

    virtual void Register(std::shared_ptr<ablate::flow::Flow>) = 0;
};

}

#endif  // ABLATELIBRARY_MONITOR_HPP
