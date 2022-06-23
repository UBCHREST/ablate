#ifndef ABLATELIBRARY_PROCESS_HPP
#define ABLATELIBRARY_PROCESS_HPP

#include <finiteVolume/finiteVolumeSolver.hpp>
namespace ablate::finiteVolume::processes {

class Process {
   public:
    virtual ~Process() = default;
    /**
     * Setup up all functions not dependent upon the mesh
     * @param fv
     */
    virtual void Setup(ablate::finiteVolume::FiniteVolumeSolver& fv) = 0;
    /**
     * Set up mesh dependent initialization
     * @param fv
     */
    virtual void Initialize(ablate::finiteVolume::FiniteVolumeSolver& fv) {};
};

}  // namespace ablate::finiteVolume::processes

#endif  // ABLATELIBRARY_PROCESS_HPP
