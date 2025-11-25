#include <AMReX.H> // for amrex::Print and amrex::Real
// #include <AMReX_MultiFab.H>
#include <aesthetics.H>
#include <iomanip>  // for std::setprecision
#include <iostream> // optional, if you use std::cout
#include <sstream>  // optional, if you later use string streams
#include <string>   // for std::string

using namespace amrex;

void PrintWelcomeMessage()
{
    amrex::Print() << " ===============================================\n";
    amrex::Print() << "        Welcome to EXAGOOP MPM Solver           \n";
    amrex::Print() << "        Developed by SAMSers at NREL            \n";
    amrex::Print() << "                 -Hari, Sree and Marc           \n";
    amrex::Print() << " ===============================================\n";
}

void PrintMessage(std::string msg, int print_length, bool begin)
{
    if (begin == true)
    {
        msg.append(print_length - msg.length(), '-');
        msg.append(1, '>');
        amrex::Print() << msg;
    }
    else
    {
        msg = " Done";
        amrex::Print() << msg;
    }
}

void PrintMessage(std::string msg, int print_length, bool begin, char c)
{
    if (begin == true)
    {
        msg.append(print_length - msg.length(), c);
        amrex::Print() << msg;
    }
    else
    {
        msg = " Done";
        amrex::Print() << msg;
    }
}

void PrintSimParams(MPMParticleContainer *mpm_pc, MPMspecs *specs)
{

    int print_length = 60;
    std::string msg = "";
    int tmpi;
    amrex::Real tmpr;

    msg = "\n     ";
    PrintMessage(msg, print_length, true, '*'); //* line

    msg = "\n     Total number of material points:";
    PrintMessage(msg, print_length, true);

    mpm_pc->Calculate_Total_Number_of_MaterialParticles(tmpi);
    amrex::Print() << " " << tmpi;

    msg = "\n     Total mass of material points:";
    PrintMessage(msg, print_length, true);

    mpm_pc->Calculate_Total_Mass_MaterialPoints(tmpr);
    amrex::Print() << " " << std::setprecision(4) << tmpr;

    msg = "\n     Total volume of material points:";
    PrintMessage(msg, print_length, true);

    mpm_pc->Calculate_Total_Vol_MaterialPoints(tmpr);
    amrex::Print() << " " << std::setprecision(4) << tmpr;

    msg = "\n     Rigid particle details:";
    PrintMessage(msg, print_length, true);

    for (int i = 0; i < specs->no_of_rigidbodies_present; i++)
    {
        msg = "\n        Total number of rigid body particles in body " +
              std::to_string(i) + ":";
        PrintMessage(msg, print_length, true);
        amrex::Print() << " " << specs->Rb[i].num_of_mp << "\n";

        msg = "\n        Total mass of rigid body " + std::to_string(i) + ":";
        PrintMessage(msg, print_length, true);
        amrex::Print() << " " << specs->Rb[i].total_mass << "\n";
    }

    msg = "\n     ";
    PrintMessage(msg, print_length, true, '*'); //* line
}
