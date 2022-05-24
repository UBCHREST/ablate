#include "runEnvironment.hpp"
#include <mpi.h>
#include <chrono>
#include <iostream>

ablate::environment::RunEnvironment::RunEnvironment() : outputDirectory(), title("") {}

ablate::environment::RunEnvironment::RunEnvironment(const parameters::Parameters& parameters, std::filesystem::path inputPath) : title(parameters.GetExpect<std::string>("title")) {
    // check to see if the output directory is set
    auto specifiedOutputDirectory = parameters.Get<std::filesystem::path>("directory");
    outputDirectory = specifiedOutputDirectory.value_or((inputPath.empty() ? std::filesystem::current_path() : inputPath.parent_path()) / title);

    // change the default value of tagDirectory, depending on if specifiedOutputDirectory was given
    auto defaultTagDirectory = !specifiedOutputDirectory.has_value();

    // Append the current time set to tag directory.
    if (parameters.Get("tagDirectory", defaultTagDirectory)) {
        // get time in milliseconds
        auto unixTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        long unixTimeLong = unixTime.count();
        long globalUnixTimeLong = unixTimeLong;

        // take the minimum time across all mpi values
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if (mpiInitialized) {
            MPI_Allreduce(&unixTimeLong, &globalUnixTimeLong, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
        }

        // back to time_point
        std::chrono::time_point<std::chrono::system_clock> startTime((std::chrono::milliseconds(globalUnixTimeLong)));
        auto startTimeStruct = std::chrono::system_clock::to_time_t(startTime);

        // As string
        std::stringstream ss;
        ss << std::put_time(std::localtime(&startTimeStruct), "_%Y-%m-%dT%H-%M-%S");

        // append to the directory
        outputDirectory += ss.str();
    }

    // have the root node copy over the input deck to the output file
    int mpiInitialized = 0;
    int mpiRank = 0;
    MPI_Initialized(&mpiInitialized);
    if (mpiInitialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    }

    if (mpiRank == 0) {
        // make the output directory
        std::filesystem::create_directories(outputDirectory);
    }
}

void ablate::environment::RunEnvironment::Setup(const ablate::parameters::Parameters& parameters, std::filesystem::path inputPath) {
    environment::RunEnvironment::runEnvironment = std::unique_ptr<environment::RunEnvironment>(new environment::RunEnvironment(parameters, inputPath));
}

void ablate::environment::RunEnvironment::RegisterCleanUpFunction(const std::string& name, std::function<void()> newFunction) {
    Get();
    auto iterator = std::find_if(runEnvironment->cleanUpFunctions.begin(), runEnvironment->cleanUpFunctions.end(), [&name](const auto& function) { return function.name == name; });
    if (iterator == runEnvironment->cleanUpFunctions.end()) {
        runEnvironment->cleanUpFunctions.push_back({.name = name, .function = newFunction});
    } else {
        iterator->function = newFunction;
    }
}

void ablate::environment::RunEnvironment::CleanUp() const {
    /* Iterate vector in reverse order */
    for (auto cleanUpFunction = cleanUpFunctions.rbegin(); cleanUpFunction != cleanUpFunctions.rend(); ++cleanUpFunction) {
        cleanUpFunction->function();
    }
}

void ablate::environment::RunEnvironment::Setup() { environment::RunEnvironment::runEnvironment = std::unique_ptr<environment::RunEnvironment>(new environment::RunEnvironment()); }
