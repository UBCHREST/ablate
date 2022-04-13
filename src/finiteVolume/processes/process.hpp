#ifndef ABLATELIBRARY_PROCESS_HPP
#define ABLATELIBRARY_PROCESS_HPP

#include <finiteVolume/finiteVolumeSolver.hpp>
namespace ablate::finiteVolume::processes {

class Process {
   public:
    virtual ~Process() = default;
    virtual void Initialize(ablate::finiteVolume::FiniteVolumeSolver& fv) = 0;
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_PROCESS_HPP
