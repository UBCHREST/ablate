#ifndef ABLATELIBRARY_HDF5OUTPUTFLOW_HPP
#define ABLATELIBRARY_HDF5OUTPUTFLOW_HPP
#include <filesystem>
#include "monitors/flow/flowMonitor.hpp"
#include "monitors/hdf5Output.hpp"
namespace ablate::monitors::flow{
class Hdf5OutputFlow: public monitors::flow::FlowMonitor, monitors::Hdf5Output{
   private:
    std::shared_ptr<ablate::flow::Flow> flow = nullptr;
    static PetscErrorCode OutputFlow(TS ts,PetscInt steps,PetscReal time,Vec u,void *mctx);

   public:
    Hdf5OutputFlow() = default;
    ~Hdf5OutputFlow() override = default;

    void Register(std::shared_ptr<ablate::flow::Flow>) override;

    PetscMonitorFunction GetPetscFunction() override{
        return OutputFlow;
    }
};
}


#endif  // ABLATELIBRARY_HDF5OUTPUTFLOW_HPP
