cmake_minimum_required(VERSION 3.18.4)

# configure cmake to use the same compilers/flags as petsc
include(config/petscCompilers.cmake)

# Set the project details
project(ablateLibrary VERSION 0.11.25)

# Load the Required 3rd Party Libaries
pkg_check_modules(PETSc REQUIRED IMPORTED_TARGET GLOBAL PETSc)

# Set the c/c++ Standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Load the cmake required libraries
include(FetchContent) # FetContent will allow downloading of other libraries
include(GNUInstallDirs) # make cache variables for install destinations

# Check if we should enable testing options
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
    include(CTest)
endif ()

# Create the library for ablate
add_library(ablateLibrary SHARED)

# Load in the subdirectories for the ablate library
add_subdirectory(src)

# find and load required libraries
include(config/cmakeFunctions.cmake)
include(config/findKokkos.cmake)
include(config/findYamlCpp.cmake)
include(config/findCppParser.cmake)
include(config/findMuParser.cmake)
include(config/findXdmfGenerator.cmake)
include(config/findTChem.cmake)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
include(config/findOpenCascade.cmake)
include(config/findJson.cmake)
include(config/findTensorFlow.cmake)
include(config/findChrestCompilerFlags.cmake)
include(config/findSLEPc.cmake)

# specify in the required libaries
target_link_libraries(ablateLibrary
        PUBLIC
        PkgConfig::PETSc
        slepc
        CHREST::cppParserLibrary
        yaml-cpp
        muparser
        CHREST::xdmfGeneratorLibrary
        TChem::tchem
        Tines::tines
        Threads::Threads
        nlohmann_json::nlohmann_json
        PRIVATE
        chrestCompilerFlags)

# Set the ablate library requirements
set_property(TARGET ablateLibrary PROPERTY CXX_EXTENSIONS ON)
set_property(TARGET ablateLibrary PROPERTY CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(TARGET ablateLibrary PROPERTY CXX_STANDARD 17)
set_property(TARGET ablateLibrary PROPERTY CXX_VISIBILITY_PRESET default)

# Setup and configure testing
if (BUILD_TESTING)
    enable_testing()
    include(GoogleTest)
    include(config/findGoogleTest.cmake)
    add_subdirectory(tests)

    # Include the format command only if we are building tests
    include(config/clangFormatter.cmake)
endif ()

# keep a separate main statement
add_executable(ablate main.cpp)
target_link_libraries(ablate PUBLIC ablateLibrary PRIVATE chrestCompilerFlags)

# include system specific hacks
include(config/systemHacks.cmake)

# install the cppParserLibrary (and others) target and create export-set
install(TARGETS ablate ablateLibrary muparser nlohmann_json cppParserLibrary yaml-cpp xdmfGeneratorLibrary chrestCompilerFlags
        EXPORT ablateTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
set_target_properties(ablate PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# install header file(s)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
update_header_paths_for_install(ablateLibrary ${PROJECT_SOURCE_DIR}/src ${CMAKE_INSTALL_INCLUDEDIR})

# generate and install export file
install(EXPORT ablateTargets
        FILE ablateTargets.cmake
        NAMESPACE CHREST::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ablate
        )

# generate the version file for the config file
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/ablateConfigVersion.cmake"
        VERSION "${CMAKE_PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
)

# create config file
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/config/Config.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/ablateConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ablate
        )

# install config files
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/ablateConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/ablateConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ablate
        )

# generate the export targets for the build tree
export(EXPORT ablateTargets
        FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/ablateTargets.cmake"
        NAMESPACE CHREST::
        )
