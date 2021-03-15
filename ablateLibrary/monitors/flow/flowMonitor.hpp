#ifndef ABLATELIBRARY_FLOW_MONITOR_HPP
#define ABLATELIBRARY_FLOW_MONITOR_HPP
#include <memory>
#include "flow/flow.hpp"
#include "monitors/monitor.hpp"

namespace ablate::monitors::flow {

class FlowMonitor : public monitors::Monitor {
   public:
    virtual ~FlowMonitor() = default;

    virtual void Register(std::shared_ptr<ablate::flow::Flow>) = 0;
};

}  // namespace ablate::monitors::flow

#endif  // ABLATELIBRARY_MONITOR_HPP
