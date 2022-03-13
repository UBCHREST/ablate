//
// Created by Owen on 3/8/2022.
//

class Irradiate { //name of the ray tracing class
    public:          //access specifier
        //CompressibleMaterial material; //This object must be replaced with the ABLATE equivalent
        const double sbc = 5.6696e-8;  // Stefan-Boltzman Constant (J/K)
        const double refTemp = 298.15;
        const double h; //This is the step size and should be set by the user input
        int Theta; //The number of angles to solve with, should be given by user input
        int Phi; //The number of angles to solve with, should be given by user input

        double radGain; //The total radiative gain for the current point.


        Irradiate() {
            //Class constructor
        }

        ~Irradiate() {
            //Class destructor
        }

       double flameIntensity(double epsilon, double temperature) {
            return epsilon * sbc * temperature * temperature * temperature * temperature / 3.1415926535897932384626433832795028841971693993;
        }

       void rayTrace() {
             //Start a timer to track progress?
             double intensity = 0; //Represents the total radiation intensity (at each point) TODO: ask, should the class be declared for a single cell or whole domain?
             //get the position of the origin TODO: this is just the current index, or iterate over every index for a whole domain.
             for(int theta = 0; theta < Theta; theta++) { //for every angle theta TODO: the number of theta iterations should be given
                //precalculate sin and cosine of the angles because they are used frequently
                for(int phi = 0; phi < Phi; phi++) {  // for every angle phi TODO: the number of phi iterations should be given
                     // direction = sin and things //create a direction vector in the current angle direction
                     // get the point of intersection with the boundary //TODO: get the index of the domain boundary and ray vector intersection point.
                     double rayIntensity = castRay(theta, phi); //need to follow the point from boundary back to the source TODO: what are the other inputs to this function?
                     intensity += rayIntensity;
                }
             }
             double kappa = 0; //TODO: How to get absorptivity coefficient here?
             radGain = kappa*intensity;
         }

        double castRay(int theta, int phi) { //This function spatially integrates the intensity over the ray vector as a function of the temperature at the current point
             double rayIntensity = flameIntensity(1, refTemp); //Initialize the ray intensity as the far field flame intensity TODO: should this be changed to the casing (boundary node) temperature in the context of a rocket?
             double intersect = 0; //TODO: Intersection point should be given here, or as input
             double magnitude = 1; //get the magnitude of the vector between the origin cell and the boundary (Set magnitude equal to intersection point minus origin)
             double exp;
             while(magnitude - h > 0) {
                 double temp = 0; //Temperature at the current point (used to calculate current point flame intensity)//TODO: Need temperature of any point in the domain
                 exp = 1; //Math.exp(-kappa[ijk[2]][ijk[1]][ijk[0]] * deltaRay) TODO: ask, what part is this exactly?
                 rayIntensity += flameIntensity(1 - exp, temp + rayIntensity*exp); //represents the flame intensity at the far field boundary? TODO: ask about this
                 //TODO: vector subtract from intersection point
                 magnitude -= h; //One step towards the origin has been completed
             }
             return rayIntensity;
         }
};