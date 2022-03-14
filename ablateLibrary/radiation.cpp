//
// Created by Owen on 3/8/2022.
//
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <math.h>

class Irradiate { //name of the ray tracing class
   public:          //access specifier, these values can be accessed outside the object itself
    ///Class Constants
    const double sbc = 5.6696e-8;  // Stefan-Boltzman Constant (J/K)
    const double refTemp = 298.15;
    const double pi = 3.1415926535897932384626433832795028841971693993;

    ///Class Inputs and Variables
    double h = 0.1; //This is the step size and should be set by the user input
    int nTheta = 100; //The number of angles to solve with, should be given by user input
    int nPhi = 100; //The number of angles to solve with, should be given by user input

    double radGain; //The total radiative gain for the current point.

    ///Class Constructors
    Irradiate() {/*Class constructor*/} //calls the default constructor and uses default parameters

    Irradiate(double spaceStep, long thetaStep, long phiStep) //calls constructor with specified number of points for discretization
        : h(spaceStep), nTheta(thetaStep), nPhi (phiStep)
    {/*Class constructor*/}

    ~Irradiate() {/*Class destructor*/}

    ///Public Class Methods
    double flameIntensity(double epsilon, double temperature) { ///Gets the flame intensity based on temperature and emissivity
        return epsilon * sbc * temperature * temperature * temperature * temperature / pi;
    }

    double rayTrace() { ///Gets the total intensity/radiative gain at a single cell
        std::cout << "Called ray tracing function. nTheta = " << nTheta << ", nPhi = " << nPhi << ", h step = " << h << "\n"; //DEBUGGING COMMENT
        double theta; //represents the actual current angle (horizontal)
        double phi; //represents the actual current angle (vertical)
        //Start a timer to track progress?
        double intensity = 0; //Represents the total radiation intensity (at each point) TODO: ask, should the class be declared for a single cell or whole domain?
        //get the position of the origin TODO: this is just the current index, or iterate over every index for a whole domain.
        for(int ntheta = 0; ntheta < nTheta; ntheta++) { //for every angle theta TODO: the number of theta iterations should be given
            //precalculate sin and cosine of the angles? because they are used frequently
            for(int nphi = 0; nphi < nPhi; nphi++) {  // for every angle phi TODO: the number of phi iterations should be given
                // direction = sin and things //create a direction vector in the current angle direction
                // get the point of intersection with the boundary //TODO: get the index of the domain boundary and ray vector intersection point.
                theta = (ntheta/nTheta)*2*pi; //converts the present angle number into a real angle
                phi = (nphi/nPhi)*2*pi; //converts the present angle number into a real angle
                double rayIntensity = castRay(theta, phi); //intensity of the single ray. Need to follow the point from boundary back to the source TODO: what are the other inputs to this function?
                intensity += rayIntensity; // *(sin(Theta)/(nTheta*nPhi)) //TODO: The additional ray intensity must be divided to give its true fractional impact (solid angle)
                std::cout << "Point Intensity: " << intensity << "\n";
            }
        }
        double kappa = 1; //Absorptivity coefficient, property of the cell? TODO: How to get absorptivity coefficient here?
        radGain = kappa*intensity; //Total energy gain of the current cell depends on absorptivity at the current cell
        return radGain;
    }

    double castRay(int theta, int phi) { ///Spatially integrates intensity over current ray based on temp & absorption at each distance
        double rayIntensity = flameIntensity(1, refTemp); //Initialize the ray intensity as the far field flame intensity TODO: should this be changed to the casing (boundary node) temperature in the context of a rocket?
        double intersect = 0; //TODO: Intersection point (coordinate) should be given here, or as input
        double magnitude = 1; //get the magnitude of the vector between the origin cell and the boundary TODO: Set magnitude equal to intersection point minus origin
        while(magnitude - h > 0) {
            double kappa = 0; //Absorptivity at the current point TODO: Need the absorptivity at any point in the domain
            double temp = 0; //Temperature at the current point (used to calculate current point flame intensity)//TODO: Need temperature of any point in the domain
            //From Java //Math.exp(-kappa[ijk[2]][ijk[1]][ijk[0]] * deltaRay) TODO: ask, what part is this exactly?
            rayIntensity += flameIntensity(1 - exp(-kappa*h), temp + rayIntensity*exp(-kappa*h)); //represents the flame intensity at the far field boundary? TODO: ask about this
            //TODO: vector subtract from intersection point to get next coordinate
            magnitude -= h; //One step towards the origin has been completed
        }
        std::cout << "rayIntensity: " << rayIntensity << "\n";
        return rayIntensity; //Final intensity of the ray as it approaches the current cell
    }
};