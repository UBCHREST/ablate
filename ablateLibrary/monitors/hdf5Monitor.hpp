#ifndef ABLATELIBRARY_HDF5MONITOR_HPP
#define ABLATELIBRARY_HDF5MONITOR_HPP
#include <petsc.h>
#include <filesystem>
#include "monitor.hpp"
#include "viewable.hpp"
namespace ablate::monitors {
class Hdf5Monitor : public Monitor {
   private:
    PetscInt index = 0;

   protected:
    PetscViewer petscViewer = nullptr;
    std::filesystem::path outputFilePath;
    const std::string extension = ".hdf5";

    std::shared_ptr<ablate::monitors::Viewable> viewableObject;

    static PetscErrorCode OutputHdf5(TS ts, PetscInt steps, PetscReal time, Vec u, void *mctx);

    const int interval;

   public:
    Hdf5Monitor(int interval = {});
    ~Hdf5Monitor() override;

    void Register(std::shared_ptr<Monitorable>) override;
    PetscMonitorFunction GetPetscFunction() override { return OutputHdf5; }
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_HDF5MONITOR_HPP
