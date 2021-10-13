#ifndef ABLATELIBRARY_PROCESS_HPP
#define ABLATELIBRARY_PROCESS_HPP

#include <finiteVolume/finiteVolume.hpp>
namespace ablate::finiteVolume::processes {

class Process {
   public:
    virtual ~Process() = default;
    virtual void Initialize(ablate::finiteVolume::FiniteVolume& fv) = 0;
};

}  // namespace ablate::flow::processes

#endif  // ABLATELIBRARY_PROCESS_HPP
