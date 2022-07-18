#include "cadFile.hpp"
#include "utilities/petscError.hpp"
#include "utilities/petscOptions.hpp"

ablate::domain::CadFile::CadFile(const std::string& nameIn, const std::filesystem::path& pathIn, std::vector<std::shared_ptr<FieldDescriptor>> fieldDescriptors, std::string generator,
                                 std::vector<std::shared_ptr<modifiers::Modifier>> modifiers, const std::shared_ptr<parameters::Parameters>& options,
                                 const std::shared_ptr<parameters::Parameters>& surfaceOptions)
    : Domain(ReadDMFromCadFile(nameIn, pathIn, surfaceOptions, generator, surfacePetscOptions, surfaceDm), nameIn, std::move(fieldDescriptors), std::move(modifiers), options) {
    // make sure that dm_refine was not set
    if (surfaceOptions) {
        if (surfaceOptions->Get("dm_refine", 0) != 0) {
            throw std::invalid_argument("dm_refine in surfaceOptions when used with ablate::domain::CadFile must be 0.");
        }
    }
}

ablate::domain::CadFile::~CadFile() {
    if (dm) {
        DMDestroy(&dm) >> checkError;
    }
    // cleanup
    if (surfacePetscOptions) {
        ablate::utilities::PetscOptionsDestroyAndCheck("ablate::domain::CadFile::ReadDMFromCadFile", &surfacePetscOptions);
    }
    if (surfaceDm) {
        DMDestroy(&surfaceDm) >> checkError;
    }
}

DM ablate::domain::CadFile::ReadDMFromCadFile(const std::string& name, const std::filesystem::path& path, const std::shared_ptr<parameters::Parameters>& surfaceOptions, const std::string& generator,
                                              PetscOptions& surfacePetscOptions, DM& surfaceDm) {
    surfacePetscOptions = nullptr;
    surfaceDm = nullptr;

    // check the path to make sure it is there
    if (!exists(path)) {
        throw std::invalid_argument("Cannot locate CAD file " + path.string());
    }

    // create a surface mesh from the cad
    DMPlexCreateFromFile(PETSC_COMM_WORLD, path.c_str(), name.c_str(), PETSC_TRUE, &surfaceDm) >> checkError;
    auto surfaceDmName = "surface_" + name;
    PetscObjectSetName((PetscObject)surfaceDm, surfaceDmName.c_str()) >> checkError;

    // if provided set the options
    if (surfaceOptions) {
        PetscOptionsCreate(&surfacePetscOptions) >> checkError;
        surfaceOptions->Fill(surfacePetscOptions);
    }
    PetscObjectSetOptions((PetscObject)surfaceDm, surfacePetscOptions) >> checkError;
    DMSetFromOptions(surfaceDm) >> checkError;

    // provide a way to view the surface mesh
    auto surfaceDmViewString = "-" + surfaceDmName + "_view";
    DMViewFromOptions(surfaceDm, nullptr, surfaceDmViewString.c_str());

    // with the surface mesh created, compute the volumetric dm
    DM dm;
    DMPlexGenerate(surfaceDm, generator.empty() ? "tetgen" : generator.c_str(), PETSC_TRUE, &dm) >> checkError;
    PetscObjectSetName((PetscObject)dm, name.c_str()) >> checkError;
    DMPlexSetRefinementUniform(dm, PETSC_TRUE) >> checkError;

    // inflate the mesh
    DMPlexInflateToGeomModel(dm) >> checkError;
    return dm;
}

#include "registrar.hpp"
REGISTER(ablate::domain::Domain, ablate::domain::CadFile, "read a cad from a file", ARG(std::string, "name", "the name of the domain/mesh object"),
         ARG(std::filesystem::path, "path", "the path to the cad file"), OPT(std::vector<ablate::domain::FieldDescriptor>, "fields", "a list of fields/field descriptors"),
         OPT(std::string, "generator", "the mesh generation package name (default is 'tetgen')"), OPT(std::vector<ablate::domain::modifiers::Modifier>, "modifiers", "a list of domain modifier"),
         OPT(ablate::parameters::Parameters, "options", "PETSc options specific to this dm.  Default value allows the dm to access global options."),
         OPT(ablate::parameters::Parameters, "surfaceOptions", "PETSc options specific to the temporary surface dm.  Default value allows the dm to access global options."));
