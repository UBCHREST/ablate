#ifndef ABLATELIBRARY_HDF5SERIALIZER_HPP
#define ABLATELIBRARY_HDF5SERIALIZER_HPP

#include <petscviewer.h>
#include <memory>
#include "serializable.hpp"
#include "serializer.hpp"
#include <filesystem>
#include <vector>

namespace ablate::io {

class Hdf5Serializer : public Serializer {

   private:
    /**
     * Private class to handle the serialization of each registered object
     */
    class Hdf5ObjectSerializer{
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
    Hdf5Serializer(int interval);

    /**
     * Handles registering the object and restore if available.
     */
    void Register(std::weak_ptr<Serializable>) override;

    // public functions to interface with the main TS
    void* GetContext() override { return this; }
    PetscSerializeFunction GetSerializeFunction() override{
        return Hdf5SerializerSaveStateFunction;
    }

    void RestoreTS(TS ts) override;
};
}

#endif  // ABLATELIBRARY_HDF5SERIALIZER_HPP
