#ifndef ABLATELIBRARY_FLOWPROCESS_HPP
#define ABLATELIBRARY_FLOWPROCESS_HPP

#include <flow/fvFlow.hpp>
namespace ablate::flow::processes {

class FlowProcess {
   public:
    virtual ~FlowProcess() = default;
    virtual void Initialize(ablate::flow::FVFlow& flow) = 0;
};

}  // namespace ablate::flow::processes

#endif  // ABLATELIBRARY_FLOWPROCESS_HPP
