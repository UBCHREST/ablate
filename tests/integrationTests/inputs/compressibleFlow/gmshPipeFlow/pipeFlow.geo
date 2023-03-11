// Gmsh project created on Tue Mar  7 08:53:01 2023
SetFactory("OpenCASCADE");
//+
Point(1) = {0.0, 0.0, 0, 1.0};
//+
Point(2) = {0.5, 0.0, 0, 1.0};
//+
Point(3) = {0.5, 0.1, 0, 1.0};
//+
Point(4) = {0.0, 0.1, 0, 1.0};
//+
Line(1) = {4, 1};
//+
Line(2) = {1, 2};
//+
Line(3) = {2, 3};
//+
Line(4) = {3, 4};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};
//+
Physical Curve("inlet", 5) = {1};
//+
Physical Curve("wall", 6) = {4, 2};
//+
Physical Curve("outlet", 7) = {3};
//+
Physical Surface("main", 8) = {1};
