#include "runEnvironment.hpp"
#include <chrono>
#include <mpi.h>
#include <iostream>

ablate::monitors::RunEnvironment::RunEnvironment(std::filesystem::path inputPath,  const parameters::Parameters& parameters):
    title(parameters.GetExpect<std::string>("title"))
{
    // check to see if the output directory is set
    auto specifiedOutputDirectory = parameters.Get<std::filesystem::path>("outputDirectory");
    outputDirectory = specifiedOutputDirectory.value_or(inputPath.parent_path() / title);

    // Append the current time set to tag directory
    if(parameters.Get("tagDirectory", true)){
        // get time in milliseconds
        auto unixTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
        long unixTimeLong = unixTime.count();
        long globalUnixTimeLong = unixTimeLong;

        // take the minimum time across all mpi values
        int mpiInitialized = 0;
        MPI_Initialized(&mpiInitialized);
        if(mpiInitialized) {
            MPI_Allreduce(&unixTimeLong, &globalUnixTimeLong, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
        }

        // back to time_point
        std::chrono::time_point<std::chrono::system_clock> startTime((std::chrono::milliseconds(globalUnixTimeLong)));
        auto startTimeStruct = std::chrono::system_clock::to_time_t(startTime);

        // As string
        std::stringstream ss;
        ss << std::put_time(std::localtime(&startTimeStruct), "_%Y-%m-%dT%X");

        // append to the directory
        outputDirectory += ss.str();
    }

    // have the root node copy over the input deck to the output file
    int mpiInitialized = 0;
    int mpiRank =0;
    MPI_Initialized(&mpiInitialized);
    if(mpiInitialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    }

    if(mpiRank  == 0){
        // make the output directory
        std::filesystem::create_directories(outputDirectory);

        // copy over the input file
        std::filesystem::copy(inputPath, outputDirectory/inputPath.filename(), std::filesystem::copy_options::overwrite_existing);
    }

}
void ablate::monitors::RunEnvironment::Setup(std::filesystem::path inputPath, const ablate::parameters::Parameters &parameters) {
    monitors::RunEnvironment::runEnvironment = std::unique_ptr<monitors::RunEnvironment>(new monitors::RunEnvironment(inputPath, parameters));
}
