//
// Created by Owen on 3/8/2022.
//
#include <iostream>
#include <math.h>
#include <vector>

///Things this code needs from ABLATE
//Location of the current cell should be known (relative to the domain)
//Get location of domain boundary based on intersection with vector
//Temperature and absorptivity(any property) of arbitrary point based on location

class Irradiate { //name of the ray tracing class
   public:          //access specifier, these values can be accessed outside the object itself
    ///Class Constants
    const double sbc = 5.6696e-8;  // Stefan-Boltzman Constant (J/K)
    const double refTemp = 298.15;
    const double pi = 3.1415926535897932384626433832795028841971693993;

    const std::vector<double> origin = {0,0,0}; //This is the point of the current cell that is being irradiated. It should be given as an input or set by the given cell.
    PetscReal test = 0;

    ///Class Inputs and Variables
    double h = 0.1; //This is the DEFAULT step size and should be set by the user input
    int nTheta = 100; //The DEFAULT number of angles to solve with, should be given by user input
    int nPhi = 100; //The DEFAULT number of angles to solve with, should be given by user input
    //TODO: Location/index of the origin cell should be known, take entire cell object as an input? If the entire domain is taken in another constructor then it will create the cells? How to get properties from coordinates.

    double radGain; //The total radiative gain for the current point.

    ///Class Constructors
    Irradiate() {/*Class constructor*/} //calls the default constructor and uses default parameters

    Irradiate(double spaceStep, long thetaStep, long phiStep) //calls constructor with specified number of points for discretization
        : h(spaceStep), nTheta(thetaStep), nPhi (phiStep)
    {/*Class constructor*/}

    //TODO: What else should be in the constructor, what other constructors should there be?

    ~Irradiate() {/*Class destructor*/}

    ///Public Class Methods
    double rayTrace() { ///Gets the total intensity/radiative gain at a single cell
        std::cout << "Called ray tracing function. nTheta = " << nTheta << ", nPhi = " << nPhi << ", h step = " << h << "\n"; //DEBUGGING COMMENT
        //Start a timer to track progress?

        double theta; //represents the actual current angle (inclination)
        double phi; //represents the actual current angle (rotation)

        double intensity = 0; //Represents the total radiation intensity (at each point) TODO: ask, should the class be declared for a single cell or whole domain?
        //get the position of the origin TODO: this is just the current index, or iterate over every index for a whole domain.

        for(int ntheta = 0; ntheta < nTheta; ntheta++) { //for every angle theta
            //precalculate sin and cosine of the angle theta? because it is used frequently
            for(int nphi = 0; nphi < nPhi; nphi++) { // for every angle phi
                theta = (ntheta/nTheta)*2*pi; //converts the present angle number into a real angle
                phi = (nphi/nPhi)*2*pi; //converts the present angle number into a real angle

                std::vector<double> direction = {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)}; //(Reference: Rad. Heat Transf. Modest pg. 11) Create a direction vector in the current angle direction TODO: All of this needs to be in Petsc data types probably? Tried and failed to import Petsc functionality. How to?
                std::vector<double> intersect = {0,0,0};// get the point of intersection with the boundary //TODO: get the index of the domain boundary and ray vector intersection point.

                double rayIntensity = castRay(theta, phi, intersect); //intensity of the single ray. Need to follow the point from boundary back to the source TODO: what are the other inputs to this function?
                intensity += rayIntensity*sin(theta)/(nTheta*nPhi); // *(sin(Theta)/(nTheta*nPhi)) //Gives the partial impact of the ray on the total sphere. The sin(theta) is a result of the polar coordinate discretization
                std::cout << "Point Intensity: " << intensity << "\n";
            }
        }
        double kappa = 1; //Absorptivity coefficient, property of the cell? TODO: How to get absorptivity coefficient here?
        radGain = kappa*intensity; //Total energy gain of the current cell depends on absorptivity at the current cell
        return radGain;
    }

    double castRay(int theta, int phi, std::vector<double> intersect) { ///Spatially integrates intensity over current ray based on temp & absorption at each distance
        std::vector<double> ray = {0,0,0}; //vector representing the ray as it is traced back from the boundary to the origin.

        double rayIntensity = flameIntensity(1, refTemp); //Initialize the ray intensity as the far field flame intensity TODO: should this be changed to the casing (boundary node) temperature in the context of a rocket?
        double magnitude = mag(intersect); //get the magnitude of the vector between the origin cell and the boundary TODO: Set magnitude equal to intersection point minus origin
        while(magnitude - h > 0) { //Keep stepping intensity through space until the origin cell has been reached
            double kappa = 0; //Absorptivity at the current point TODO: Need the absorptivity at any point in the domain
            double temp = 0; //Temperature at the current point (used to calculate current point flame intensity)//TODO: Need temperature of any point in the domain
            //From Java, implemented in line below //Math.exp(-kappa[ijk[2]][ijk[1]][ijk[0]] * deltaRay) TODO: ask, what part is this exactly?
            rayIntensity += flameIntensity(1 - exp(-kappa*h), temp + rayIntensity*exp(-kappa*h)); //represents the flame intensity at the far field boundary? TODO: ask about this
            ray = 0; //std::Subtract(intersect,origin);//TODO: vector subtract from intersection point to get next coordinate, how to identify point at which properties are measured?
            magnitude -= h; //One step towards the origin has been completed
        }
        std::cout << "rayIntensity: " << rayIntensity << "\n";
        return rayIntensity; //Final intensity of the ray as it approaches the current cell
    }

    double flameIntensity(double epsilon, double temperature) { ///Gets the flame intensity based on temperature and emissivity
        return epsilon * sbc * temperature * temperature * temperature * temperature / pi;
    }

    double mag(std::vector<double> vector) { ///Simple function to find magnitude of a vector
        double magnitude = 0;
        for (const int& i : vector) { //Sum of all points
            magnitude += i*i; //Squared
        }
        magnitude = sqrt(magnitude); //Square root of the resulting sum
        return magnitude; //Return the magnitude of the vector as a double
    }
};