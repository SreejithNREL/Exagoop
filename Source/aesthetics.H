#ifndef AESTHETICS_H_
#define AESTHETICS_H_

#include <mpm_particle_container.H>

// using namespace std;
// using namespace amrex;

void PrintWelcomeMessage();

void PrintMessage(std::string msg, int print_length, bool begin);

void PrintMultiLineMessage(std::string msg, int print_length, bool begin);

void PrintMessage(std::string msg, int print_length, bool begin, char c);

void PrintSimParams(MPMParticleContainer *mpm_pc, MPMspecs *specs);

#endif
