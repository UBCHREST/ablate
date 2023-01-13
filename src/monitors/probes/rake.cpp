#include "rake.hpp"
#include <fstream>
#include <utility>
#include "utilities/mathUtilities.hpp"
#include "utilities/mpiUtilities.hpp"

ablate::monitors::probes::Rake::Rake(std::string name, std::vector<double> start, std::vector<double> end, int number)
    : rakeName(std::move(name)), rakePath(ablate::environment::RunEnvironment::Get().GetOutputDirectory() / rakeName) {
    if (start.size() != end.size()) {
        throw std::invalid_argument("The start and end vectors must be same length.");
    }
    if (number < 2) {
        throw std::invalid_argument("At least two probe locations in a rake are required.");
    }

    // Make the output directory
    std::filesystem::create_directories(rakePath);

    // determine the normal direction and delta
    std::vector<double> direction(start.size());
    utilities::MathUtilities::Subtract(start.size(), end.data(), start.data(), direction.data());

    // Get the length and unit vector
    double length = utilities::MathUtilities::MagVector(direction.size(), direction.data());
    utilities::MathUtilities::NormVector(direction.size(), direction.data());
    double delta = length / (number - 1);

    // March over the start and end to determine number of points
    for (int p = 0; p < number; p++) {
        // The new point will be
        std::vector<PetscReal> probeLoc(start);
        for (std::size_t c = 0; c < direction.size(); c++) {
            probeLoc[c] += delta * p * direction[c];
        }

        // store the probe name
        auto probeName = rakeName + "." + std::to_string(p);

        // Create the probe
        list.emplace_back(probeName, probeLoc);
    }
}

void ablate::monitors::probes::Rake::Report(MPI_Comm comm) const {
    // Get the rank
    PetscMPIInt rank;
    MPI_Comm_rank(comm, &rank) >> utilities::MpiUtilities::checkError;
    if (rank == 0) {
        std::ofstream probeFile;
        probeFile.open(GetDirectory() / (rakeName + ".txt"));
        std::string header = "p ";
        switch (list.front().location.size()) {
            case 3:
                header += " x y z";
                break;
            case 2:
                header += " x y";
                break;
            case 1:
                header += " x";
                break;
        }
        probeFile << header << std::endl;
        for (std::size_t p = 0; p < list.size(); p++) {
            probeFile << p;
            for (const auto& c : list[p].location) {
                probeFile << " " << c;
            }
            probeFile << std::endl;
        }

        probeFile.close();
    }
}

#include "registrar.hpp"
REGISTER(ablate::monitors::probes::ProbeInitializer, ablate::monitors::probes::Rake, "Inserts probes along a line", ARG(std::string, "name", "The rake name"),
         ARG(std::vector<double>, "start", "the starting point for the rake"), ARG(std::vector<double>, "end", "the ending point for the rake"),
         ARG(int, "number", "the number of probes in the rake"));
