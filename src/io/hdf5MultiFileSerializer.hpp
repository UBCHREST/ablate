#ifndef ABLATELIBRARY_HDF5MULTIFILESERIALIZER_HPP
#define ABLATELIBRARY_HDF5MULTIFILESERIALIZER_HPP

#include <petscviewer.h>
#include <filesystem>
#include <io/interval/interval.hpp>
#include <memory>
#include <vector>
#include "parameters/parameters.hpp"
#include "serializable.hpp"
#include "serializer.hpp"
#include "utilities/loggable.hpp"

namespace ablate::io {

class Hdf5MultiFileSerializer : public Serializer, private utilities::Loggable<Hdf5MultiFileSerializer> {
   private:
    // Use the interval class to determine when to write to file
    const std::shared_ptr<ablate::io::interval::Interval> interval;

    // file extension
    inline const static std::string extension = ".hdf5";

    // keep track of time and increments
    PetscReal time;
    PetscReal dt;
    PetscInt sequenceNumber;
    PetscInt timeStep;
    bool resumed = false;

    // keep a list of postProcesses ids
    std::vector<std::string> postProcessesIds;

    // Hold the pointer to each serializable object;
    std::vector<std::weak_ptr<Serializable>> serializables;

    // an optional petscOptions that is used for this solver
    PetscOptions petscOptions = nullptr;

    //! Petsc function used to save the system state
    static PetscErrorCode Hdf5MultiFileSerializerSaveStateFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

    //! Private functions to load and save the ts metadata data
    void SaveMetadata(TS ts) const;

    //! Private functions to load and save the ts metadata data
    [[nodiscard]] std::filesystem::path GetOutputFilePath(const std::string& objectId) const;

    //! private function to get the output directory
    static std::filesystem::path GetOutputDirectoryPath(const std::string& objectId);

   public:
    /**
     * Separates into multiple files to solve some io issues
     */
    explicit Hdf5MultiFileSerializer(std::shared_ptr<ablate::io::interval::Interval>, std::shared_ptr<parameters::Parameters> options = nullptr);

    /**
     * Allow file cleanup
     */
    ~Hdf5MultiFileSerializer() override;

    /**
     * Handles registering the object and restore if available.
     */
    void Register(std::weak_ptr<Serializable>) override;

    //! public functions to interface with the main TS
    void* GetContext() override { return this; }

    //! public functions to interface with the main TS
    PetscSerializeFunction GetSerializeFunction() override { return Hdf5MultiFileSerializerSaveStateFunction; }

    void RestoreTS(TS ts) override;

    /**
     * Restores a specific sequence number from the collection of output files in the directory.
     * @param sequenceNumber
     */
    void RestoreFromSequence(PetscInt currentSequenceNumber, std::weak_ptr<Serializable> serializable);
};

}  // namespace ablate::io
#endif  // ABLATELIBRARY_HDF5MULTIFILESERIALIZER_HPP
