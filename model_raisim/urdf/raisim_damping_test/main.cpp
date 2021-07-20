#include <iostream>
#include "raisim/World.hpp"
#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <time.h>

using std::cout; using std::ofstream;
using std::endl; using std::string;
using std::fstream;
#define OUT_FILE false
#define SWEEP false

using namespace std;

int main(int argc, char *argv[]) {

    // Hyper Parameter defination
     float inertia = 6*6*103E-6;     // joint inertia
//     float inertia = 0.008966;     // joint inertia
//    float inertia = 0.003708;
    float kp = 80.0f;//80.0f;           // joint stiffness
    float kd = 1.0f;//1.0f;            // joint damping
    float omega = 8.0f;//10.0f;        // pTarget swing frequency
    float dt = 2e-3;            // time step of calculation
//    float total_time = 15.0f * 10.0f;    // total simulation time
    float total_time = 20.0;    // total simulation time

//    raisim::World::setActivationKey(raisim::Path("./activation.raisim").getString());
    raisim::World::setActivationKey(raisim::Path("activation.raisim").getString());
    raisim::World world;

    Eigen::VectorXd jointDgain;
    Eigen::VectorXd jointPgain;
    Eigen::VectorXd Ptarget;
    Eigen::VectorXd Dtarget;
    Eigen::VectorXd gc_;
    Eigen::VectorXd gv_;
    raisim::VecDyn RotorInertia_;
    RotorInertia_.setZero(1);
    RotorInertia_[0] = inertia;

    time_t timep;

    auto nunchucks = world.addArticulatedSystem("test.urdf");
//    auto nunchucks = world.addArticulatedSystem("BlackPanther_2d.urdf");
    nunchucks->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

    // nunchucks->setMass(2, inertia);
    nunchucks->setRotorInertia(RotorInertia_);
    nunchucks->updateMassInfo();
    cout << nunchucks->getMassMatrix() <<endl;
    world.setTimeStep(dt);

    int gcDim_ = nunchucks->getGeneralizedCoordinateDim();
    int gvDim_ = nunchucks->getDOF();

    jointDgain.setZero(gvDim_);
    jointPgain.setZero(gvDim_);
    gc_.setZero(gcDim_);
    gv_.setZero(gvDim_);
    Ptarget.setZero(gcDim_);
    Dtarget.setZero(gvDim_);

    jointDgain.tail(1) << kd;
    jointPgain.tail(1) << kp;

    nunchucks->setPdGains(jointPgain, jointDgain);
    nunchucks->setPdTarget(Ptarget, Dtarget);
    nunchucks->getState(gc_, gv_);
    gv_.tail(1) << 0;
    gc_.tail(1) << 0;
    nunchucks->setState(gc_, gv_);

    float current_time = 0.0;
    float pTarget_last = 0.0f;
    float pTarget_next = 0.0f;
    float ratio = 0.0;

#if SWEEP
#if OUT_FILE
    time(&timep);
    char filename[256] = {0};
    strftime( filename, sizeof(filename), "%Y.%m.%d %H-%M-%S.xls",localtime(&timep) );
    fstream file_out;
//    file_out.open("/home/wooden/Desktop/0322-0328_research_plan/data/"+std::string(filename), std::ios_base::out);
    file_out.open("/home/wooden/Desktop/0329-0411/data/"+std::string(filename), std::ios_base::out);
    file_out << "des_q0"<< "\t" << "raw_q0" << "\t" << "raw_dq0" << endl;
    for(int i=0; i<int(total_time/dt); i++){
        // omega = int(current_time/15.0f)+1.0f;  // change omega
        omega = 8;
        Ptarget.tail(1) << sin(2*3.1415926*omega*floor(current_time/0.001)*0.001);
        nunchucks->setPdTarget(Ptarget, Dtarget);
        nunchucks->setPdGains(jointPgain, jointDgain);
        world.integrate();
        file_out << Ptarget.tail(1) << "\t" << nunchucks->getGeneralizedCoordinate() << "\t" << nunchucks->getGeneralizedVelocity() << endl;
        current_time += dt;
    }
    file_out.close();
#else

#endif

#else
#if OUT_FILE
    time(&timep);
    char filename[256] = {0};
    strftime( filename, sizeof(filename), "%Y.%m.%d %H-%M-%S.txt",localtime(&timep) );
    fstream file_out;
    file_out.open("/home/wooden/Desktop/0322-0328_research_plan/data/"+std::string(filename), std::ios_base::out);
    file_out << "inertia " << inertia << " kp " << kp << " kd " << kd << " omega " << omega << " dt " << dt << endl;
    for (int i = 0; i < int(total_time / dt); i++) {
        file_out << "t: " << current_time << " q " << nunchucks->getGeneralizedCoordinate() << " dq " << nunchucks->getGeneralizedVelocity() << endl;
        // Ptarget.tail(1) << sin(2*3.14*omega*current_time);
        Ptarget.tail(1) << sin(2*3.1415926*omega*floor(current_time/0.002)*0.002);
//        if(fmod(current_time,0.002)<1e-6){
//            pTarget_last = pTarget_next;
//            pTarget_next = sin(2*3.1415926*omega*floor(current_time/0.002)*0.002);
//        }
//        ratio = fmod(current_time,0.002)/0.002;
//        Ptarget.tail(1) <<  ratio*pTarget_next+(1.0-ratio)*pTarget_last;
        nunchucks->setPdTarget(Ptarget, Dtarget);
        nunchucks->setPdGains(jointPgain, jointDgain);
        world.integrate();
        current_time += dt;
//        cout << nunchucks->getGeneralizedCoordinate() << " " << (nunchucks->getGeneralizedForce()) << endl;
//        cout << nunchucks->getMassMatrix() <<endl;
    }
    file_out.close();

    // M(u+-u-)=dt(-kd((u++u-)/2-u_ref))
    //
#else
    for (int i = 0; i < int(total_time / dt); i++) {
//        file_out << "t: " << current_time << " q " << nunchucks->getGeneralizedCoordinate() << " dq " << nunchucks->getGeneralizedVelocity() << endl;
        // Ptarget.tail(1) << sin(2 * 3.14 * omega * current_time);
        Ptarget.tail(1) << sin(2*3.1415926*omega*floor(current_time/0.002)*0.002);
        nunchucks->setPdTarget(Ptarget, Dtarget);
        nunchucks->setPdGains(jointPgain, jointDgain);
        world.integrate();
        current_time += dt;
//        cout << "target" << Ptarget.tail(1) <<
//        " " << nunchucks->getGeneralizedCoordinate() <<
//        " " << (nunchucks->getGeneralizedVelocity()) <<
//        " "<< nunchucks->getGeneralizedForce()<<endl;
//        cout << nunchucks->getMassMatrix() <<endl;
    }
#endif
#endif


    std::cout << "Finished!" << std::endl;
    return 0;
}
