#ifndef ABLATELIBRARY_HDF5SERIALIZER_HPP
#define ABLATELIBRARY_HDF5SERIALIZER_HPP

#include <petscviewer.h>
#include <filesystem>
#include <io/interval/interval.hpp>
#include <memory>
#include <vector>
#include "serializable.hpp"
#include "serializer.hpp"
#include "utilities/loggable.hpp"

namespace ablate::io {

class Hdf5Serializer : public Serializer {
   private:
    /**
     * Private class to handle the serialization of each registered object
     */
    class Hdf5ObjectSerializer : private utilities::Loggable<Hdf5Serializer> {
       private:
        PetscViewer petscViewer = nullptr;
        const std::weak_ptr<Serializable> serializable;

        inline const static std::string extension = ".hdf5";
        std::filesystem::path filePath;

       public:
        explicit Hdf5ObjectSerializer(std::weak_ptr<Serializable> serializable, PetscInt sequenceNumber, PetscReal time, bool resumed);
        ~Hdf5ObjectSerializer();

        void Save(PetscInt sequenceNumber, PetscReal time);
    };

    // Use the interval class to determine when to write to file
    const std::shared_ptr<ablate::io::interval::Interval> interval;

    // keep track of time and increments
    PetscReal time;
    PetscReal dt;
    PetscInt sequenceNumber;
    PetscInt timeStep;
    bool resumed = false;

    // Hold the pointer to each serializers;
    std::vector<std::unique_ptr<Hdf5ObjectSerializer>> serializers;

    // Petsc function used to save the system state
    static PetscErrorCode Hdf5SerializerSaveStateFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* mctx);

    // Private functions to load and save the ts metadata data
    void SaveMetadata(TS ts);

   public:
    explicit Hdf5Serializer(std::shared_ptr<ablate::io::interval::Interval>);

    /**
     * Handles registering the object and restore if available.
     */
    void Register(std::weak_ptr<Serializable>) override;

    // public functions to interface with the main TS
    void* GetContext() override { return this; }
    PetscSerializeFunction GetSerializeFunction() override { return Hdf5SerializerSaveStateFunction; }

    void RestoreTS(TS ts) override;
};
}  // namespace ablate::io

#endif  // ABLATELIBRARY_HDF5SERIALIZER_HPP
