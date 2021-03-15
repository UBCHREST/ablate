#ifndef ABLATELIBRARY_HDF5OUTPUT_HPP
#define ABLATELIBRARY_HDF5OUTPUT_HPP
#include <petsc.h>
#include <filesystem>
namespace ablate::monitors {
class Hdf5Output {
   protected:
    PetscViewer petscViewer = nullptr;
    std::filesystem::path outputFilePath;
    const std::string extension = ".hdf5";

   public:
    Hdf5Output() = default;
    virtual ~Hdf5Output();
};
}  // namespace ablate::monitors

#endif  // ABLATELIBRARY_HDF5OUTPUT_HPP
