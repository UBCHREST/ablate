#ifndef ABLATELIBRARY_HDF5OUTPUT_HPP
#define ABLATELIBRARY_HDF5OUTPUT_HPP
#include "monitors/flow/monitor.hpp"

namespace ablate::monitors::flow{
class Hdf5Output: public monitors::flow::Monitor{
   private:
    std::shared_ptr<ablate::flow::Flow> flow = nullptr;
    PetscViewer petscViewer = nullptr;

    static PetscErrorCode OutputFlow(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

   public:
    Hdf5Output() = default;
    ~Hdf5Output() override;

    void Register(std::shared_ptr<ablate::flow::Flow>) override;

    PetscMonitorFunction GetPetscFunction() override{
        return OutputFlow;
    }
};
}


#endif  // ABLATELIBRARY_HDF5OUTPUT_HPP
