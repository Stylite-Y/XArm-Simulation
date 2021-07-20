//
// BlackPanther Trot Pattern Reinforcement Learning Environment
// V5 mimic wbic controller
// Author: Wooden Jin, yongbinjin@zju.edu.cn
// CopyRight @ Xmech, ZJU http://www.xmech.zju.edu.cn/
// Ref: https://github.com/robotlearn/raisimpy;
//      https://github.com/bulletphysics/bullet3
//      https://github.com/google-research/motion_imitation

/* Convention
*
*   observation space = [ cmd                  n =  3, si =  0
*                         phase                n =  2, si =  3
*                         theta                n = 12, si =  5
*                         theta_dot            n = 12, si = 17
*                         posture              n =  3, si = 29
*                         omega                n =  3, si = 33
*
*   ref's feature  = [  theta      n = 12, si = 0
*                       theta_dot  n = 12, si = 12
*                       z          n = 1,  si = 24
*                       phase      n = 2,  si = 25
*                       cmd        n = 3,  si = 27]
*
*/

#include <stdlib.h>
#include <cstdint>
#include <set>
#include <random>
#include <iomanip>
#include <iostream>
#include <utility>
#include <math.h>

#include "raisim/OgreVis.hpp"
#include "RaisimGymEnv.hpp"
#include "visSetupCallback.hpp"

#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
// #include "visualizer/raisimBasicImguiPanel.hpp"
#include "visualizer/raisimCustomerImguiPanel.hpp"
#include "VectorizedEnvironment.hpp"

#define PI 3.1415926
#define MAGENTA "\033[35m"           /* Magenta */
#define BOLDCYAN "\033[1m\033[36m"   /* Bold Cyan */
#define BOLDYELLOW "\033[1m\033[33m" /* Bold Yellow */
#define DEBUG0 0
#define DEBUG1 0
#define DEBUG2 0

using std::default_random_engine;
using std::normal_distribution;

/*
 * produce a circle position in xy plane
 * */
inline Eigen::Vector3d circle_place(double radius, unsigned int index, unsigned int num, int z) {
    double temp_angle = 0.0;
    temp_angle = index * 1.0 / num * 2 * PI;
    return Eigen::Vector3d(radius * sin(temp_angle), radius * cos(temp_angle), z);
}

/*
 * reshape the sampling prob density
 */
inline double sampling_reshape(double ratio) {
    if (ratio < 0.5 && ratio > 0) {
        return ratio * 4.0 / 3.0;
    } else {
        return (2.0 * ratio + 1.0) / 3.0;
    }
}

/*
 * using cubicBezier to shape the trajectory of foot end, which can promise small clash during contact ground
 */
inline Eigen::Vector3d cubicBezier(Eigen::Vector3d p0, Eigen::Vector3d pf, double phase) {
    Eigen::Vector3d pDiff = pf - p0;
    double bezier = phase * phase * phase + 3.0 * (phase * phase * (1.0 - phase));
    return p0 + bezier * pDiff;
}

/*
 * using gauss function to shape the trajectory of foot end
 */
inline double gauss(double x, double width, double height) {
    return height * exp(-(x - width / 2) * (x - width / 2) / (2 * (width / 6) * (width / 6)));
}

/*
 * smooth traj
 */
inline Eigen::VectorXd Bezier2(Eigen::Vector3d p0, Eigen::Vector3d pf, double phase, double height) {
    Eigen::Vector3d p;
    double bezier = phase * phase * phase + 3.0 * (phase * phase * (1.0 - phase));
    p << p0[0] + bezier * (pf[0] - p0[0]),
            p0[1] + bezier * (pf[1] - p0[1]),
            //p0[2] + sin(phase * PI) * height;
            p0[2] + gauss(phase, 1.0, height);
    return p;
}

/*
 * Motor model
 */
#define motor_kt 0.05
#define motor_R 0.173
#define motor_tau_max 3.0
#define motor_battery_v 24
#define motor_damping 0.01
#define motor_friction 0.2
double gear_ratio[12] = {6, 6, 9.33, 6, 6, 9.33, 6, 6, 9.33, 6, 6, 9.33};

inline double sgn(double x) {
    if (x > 0) { return 1.0; }
    else { return -1.0; }
}

/*
 * simplified motor model
 * */
inline void
RealTorque(Eigen::Ref<Eigen::VectorXd> torque, Eigen::Ref<Eigen::VectorXd> qd, int motor_num, bool isFrictionEnabled) {
    double tauDesMotor = 0.0;    // desired motor torque
    double iDes = 0.0;           // desired motor iq
    double bemf = 0.0;           // Back electromotive force
    double vDes = 0.0;           // desired motor voltage
    double vActual = 0.0;        // real motor torque
    double tauActMotor = 0.0;    // real motor torque

    for (int i = 0; i < motor_num; i++) {
        tauDesMotor = torque[i] / gear_ratio[i];
        iDes = tauDesMotor / (motor_kt * 1.5);
        bemf = qd[i] * gear_ratio[i] * motor_kt * 2;
        vDes = iDes * motor_R + bemf;
        vActual = fmin(fmax(vDes, -motor_battery_v), motor_battery_v);
        tauActMotor = 1.5 * motor_kt * (vActual - bemf) / motor_R;
        torque[i] = gear_ratio[i] * fmin(fmin(-motor_tau_max, tauActMotor), motor_tau_max);
        if (isFrictionEnabled) {
            torque[i] = torque[i] - motor_damping * qd[i] - motor_friction * sgn(qd[i]);
        }
    }
}

/*
void readCSV_m2(const std::string &w_file, Eigen::MatrixXf *p) {

    std::ifstream in(w_file);
    std::string line;
    int row = 0;
    int col = 0;

    Eigen::MatrixXf res(1, 1);
    Eigen::RowVectorXf rowVector(1);
    if (in.is_open()) {

        while (std::getline(in, line, '\n')) {

            // cout << line << endl;

            char *ptr = (char *) line.c_str();
            int len = line.length();

            col = 0;

            char *start = ptr;

            for (int i = 0; i < len; i++) {

                if (ptr[i] == ',') {
                    //res(row, col++) = atof(start);
                    rowVector(col++) = atof(start);
                    start = ptr + i + 1;
                    rowVector.conservativeResize(col + 1);
                }
            }
            //res(row++, col) = atof(start);
            rowVector(col) = atof(start);
            res.conservativeResize(row + 1, col + 1);
            res.row(row++) = rowVector;
        }

        in.close();
    } else {
        std::cout << "Can Not Load Parameter File of " << w_file << std::endl;
    }

    p = &res;
    // return res;
}

Eigen::MatrixXf readCSV_m(const std::string &w_file) {

    std::ifstream in(w_file);
    std::string line;
    int row = 0;
    int col = 0;

    Eigen::MatrixXf res(1, 1);
    Eigen::RowVectorXf rowVector(1);
    if (in.is_open()) {

        while (std::getline(in, line, '\n')) {

            // cout << line << endl;

            char *ptr = (char *) line.c_str();
            int len = line.length();

            col = 0;

            char *start = ptr;

            for (int i = 0; i < len; i++) {

                if (ptr[i] == ',') {
                    //res(row, col++) = atof(start);
                    rowVector(col++) = atof(start);
                    start = ptr + i + 1;
                    rowVector.conservativeResize(col + 1);
                }
            }
            //res(row++, col) = atof(start);
            rowVector(col) = atof(start);
            res.conservativeResize(row + 1, col + 1);
            res.row(row++) = rowVector;
        }

        in.close();
    } else {
        std::cout << "Can Not Load Parameter File of " << w_file << std::endl;
    }

    return res;
}
*/


//extern const std::string ref_file;
//Eigen::MatrixXf ref;


namespace raisim {
    class ENVIRONMENT : public RaisimGymEnv {

    public:

        explicit ENVIRONMENT(const std::string &resourceDir,
                             const YAML::Node &cfg,
                             bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), distribution_(0.0, 0.2), visualizable_(visualizable) {

            // load parameter from yaml file
            parameter_load_from_yaml(cfg);

#if DEBUG0
            std::cout << "11" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            // ------------------------------------------------------------
            // since now, suppose the hoof is soft and stick enough
//             world_->setDefaultMaterial(2.0, 0.0, 10.0);  // set default material

            // create world, add robot and create ground
            // load black panther URDF and create dynamics model

            minicheetah_ = world_->addArticulatedSystem(resourceDir_ + "/black_panther.urdf");
            minicheetah_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            world_->setERP(0, 0);

            // set ground of environment
            HeightMap *hm;
            Ground *ground;

            if (flag_terrain) {
                raisim::TerrainProperties terrainProperties;
                terrainProperties.frequency = 1;
                terrainProperties.zScale = 0.1;
                terrainProperties.xSize = 20.0;
                terrainProperties.ySize = 20.0;
                terrainProperties.xSamples = 500;
                terrainProperties.ySamples = 500;
                terrainProperties.fractalOctaves = 3;
                terrainProperties.fractalLacunarity = 2.0;
                terrainProperties.fractalGain = 0.25;
                hm = world_->addHeightMap(0.0, 0.0, terrainProperties);
            } else {
                ground = world_->addGround();
            }

            // -------------------------------------------------------------
            // init crucial learning
            if (flag_crucial) {
                for (int i = 0; i < num_cube; i++) {
                    cubes.emplace_back(world_->addBox(cube_len, cube_len, cube_len, cube_mass));
                    cubes.back()->setPosition(circle_place(cube_place_radius, i, num_cube, 2.0));
                    // do not calculate dynamics at the very begin
                    cubes.back()->setBodyType(raisim::BodyType::STATIC);
                }
            }

            // -------------------------------------------------------------
            // get robot data
            gcDim_ = minicheetah_->getGeneralizedCoordinateDim();
            gvDim_ = minicheetah_->getDOF();
            nJoints_ = 12;  // 12 joints dof

            // iniyialize containers
            gc_.setZero(gcDim_);            // generalized dof, 3 position, 4 quaternion, 12 joint
            gc_init_.setZero(gcDim_);       // store the initial robot state
            gv_.setZero(gvDim_);            // generalized velocity, 3 translate, 3 rotation, 12 joint
            gv_init_.setZero(gvDim_);       // store the initial robot velocity
            // target define, ref position, velocity and torque_ff
            torque_.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget12_.setZero(nJoints_);
            pTarget12Last_.setZero(nJoints_);
            jointLast_.setZero(nJoints_);

            // normal configuration of MiniCheetah
            gc_init_ << 0, 0, 0.31,       // position_x,y,z of minicheetah_center
                    1, 0, 0, 0,        // posture_w,x,y,z of minicheetah body
                    -abad_, -0.78, 1.57,  // Font right leg, abad, hip, knee
                    abad_, -0.78, 1.57,   // Font left leg, abad, hip, knee
                    -abad_, -0.78, 1.57,  // hind right leg, abad, hip, knee
                    abad_, -0.78, 1.57;  // hind left leg, abad, hip, knee

            random_init.setZero(gcDim_);
            random_vel_init.setZero(gvDim_);
            // random_init.setConstant(gcDim_, 1.0);  // for random init
            random_init.tail(19) = gc_init_.tail(19);
            EndEffector_.setZero(nJoints_);
            EndEffectorRef_.setZero(nJoints_);
            EndEffectorOffset_.setZero(nJoints_);
            EndEffectorOffset_ << 0.19, -0.058, 0,       // FR
                    0.19, 0.058, 0,       // FL
                    -0.19, -0.058, 0,       // HR
                    -0.19, 0.058, 0;       // HL

            // set pd gains
            // Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero(gvDim_);
            // jointPgain.tail(nJoints_).setConstant(stiffness);
            jointPgain.tail(nJoints_) << stiffness * abad_ratio, stiffness, stiffness,
                    stiffness * abad_ratio, stiffness, stiffness,
                    stiffness * abad_ratio, stiffness, stiffness,
                    stiffness * abad_ratio, stiffness, stiffness;
            // jointPgain.setConstant(80.0);
            jointDgain.setZero(gvDim_);
            // jointDgain.tail(nJoints_).setConstant(0.2);
            // jointDgain.setConstant(1.0);
            // jointDgain.tail(nJoints_).setConstant(damping);
            jointDgain.tail(nJoints_) << damping * abad_ratio, damping, damping,
                    damping * abad_ratio, damping, damping,
                    damping * abad_ratio, damping, damping,
                    damping * abad_ratio, damping, damping;
            torque.setZero(nJoints_);
            torque_last.setZero(nJoints_);
            torque_limit.setZero(nJoints_);
            torque_limit << 18, 18, 27, 18, 18, 27, 18, 18, 27, 18, 18, 27;

            minicheetah_->setPdGains(jointPgain, jointDgain);
            minicheetah_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));  // no ff torque and outside force

            // ----------------------------------------------------------------------------
            // set observation and action space
            obDim_ = 35;
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);
            obMean_.setZero(obDim_);
            obStd_.setZero(obDim_);
            obDouble_.setZero(obDim_);
            obDouble_last_.setZero(obDim_);
            obScaled_.setZero(obDim_);

            // action & observation scaling
            actionMean_ = gc_init_.tail(nJoints_);   // read 12 data from the back
            // actionStd_.setConstant(0.6);
            actionStd_.setConstant(1.0);
            // actionStd_.setConstant(10.0);          // needed to be checked
            // Mean value of observations
            obMean_ << (Vx_max + Vx_min) / 2, (Vy_max + Vy_min) / 2, (omega_max + omega_min) / 2,
                    0.0, 0.0,
                    gc_init_.tail(12),
                    Eigen::VectorXd::Constant(12, 0.0),
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0;
            // cmd, vel_x_cmd[Vx_min~Vx_max] ...
            // Standard deviation of observations
            obStd_ << 1.0, 1.0, 1.0,
                    1.0, 1.0,
                    Eigen::VectorXd::Constant(12, 1.0 / 1.0),
//                    Eigen::VectorXd::Constant(12, 5.0),
                    5.0, 35.0, 40.0, 5.0, 35.0, 40.0, 5.0, 35.0, 40.0, 5.0, 35.0, 40.0,
                    0.7, 0.7, 0.7,
                    3.0, 3.0, 3.0;
//                    2.0, 2.0, 2.0;

            // ---------------------------------------------------------------------
            max_len = sqrt(l_hip_ * l_hip_ + (l_calf_ + l_thigh_) * (l_calf_ + l_thigh_));
            filter_para = (flag_filter) ? (1 - freq * control_dt_) : 0;    // calculate filter parameter
            phase_.setZero(4);
            phase_ << 0.5, 0.0, 0.0, 0.5;
            // init the joint ref
            jointRef_.setZero(nJoints_);
            jointRefLast_.setZero(nJoints_);
            jointDotRef_.setZero(nJoints_);
            jointRef_ << -abad_, 0.0, 0.0,
                    abad_, 0.0, 0.0,
                    -abad_, 0.0, 0.0,
                    abad_, 0.0, 0.0;

            ForwardDirection_.setZero(4);
            ForwardDirection_ << 1, 0, 0, 0;

            if (flag_ObsFilter) {
                ObsFilterAlpha =
                        2.0 * 3.14 * control_dt_ * ObsFilterFreq / (2.0 * 3.14 * control_dt_ * ObsFilterFreq + 1.0);
            }

            // -------------------------------------------------------------------------------------------------
            // disturb the dynamics of robot
            // world_->setDefaultMaterial(4.0,0.2,5.0);
            if (flag_StochasticDynamics) {
                // disturb material
//                world_->setDefaultMaterial(rand() / float(RAND_MAX) * 4.0, rand() / float(RAND_MAX) * 1.0,
//                                           rand() / float(RAND_MAX) * 5.0);
                world_->setDefaultMaterial(rand() / float(RAND_MAX) * 2.0 + 0.4,
                                           rand() / float(RAND_MAX) * 0.6,
                                           rand() / float(RAND_MAX) * 10.0);

                // disurb mass
                double temp___ = 0.0;
                for (size_t i = 0; i < size_t(13); i++) {
                    temp___ = double((rand() / float(RAND_MAX) - 0.5) / 0.5 * mass_distrubance_ratio + 1.0);
                    double temp_mass = minicheetah_->getMass(i);
                    minicheetah_->getMass()[i] = temp_mass * temp___;
                }
                // disturb mass com
                Eigen::VectorXd com_noise;
                com_noise.setRandom(3);
                for (size_t i = 0; i < size_t(13); i++) {
                    com_noise.setRandom(3);
                    com_noise *= com_distrubance;
                    minicheetah_->getLinkCOM()[i].e() += com_noise;
                }
                minicheetah_->updateMassInfo();
                temp___ = double((rand() / float(RAND_MAX) - 0.5) / 0.5 * calf_distrubance);
                minicheetah_->getJointPos_P()[3].e()[2] += temp___;
                minicheetah_->getJointPos_P()[6].e()[2] += temp___;
                minicheetah_->getJointPos_P()[9].e()[2] += temp___;
                minicheetah_->getJointPos_P()[12].e()[2] += temp___;
            }

            // gui::rewardLogger.init({"forwardVelReward", "torqueReward", "patternReward", "DeepMimicReward"});
            gui::rewardLogger.init({"EndEffector",
                                    "HeightKeep",
                                    "BalanceKeep",
                                    "Joint",
                                    "JointDot",
                                    "Velocity",
                                    "Torque",
                                    "Forward",
                                    "lateral",
                                    "yaw"});
            raisim::gui::showContacts = true;  // always show the contact points
            // raisim::gui::showForces = true;    // always show the contact forces
            // std::cout<<"MINI_118"<<std::endl;
            // visualize if it is the efirst environment
            if (visualizable_) {
                auto vis = raisim::OgreVis::get();

                /// these method must be called before initApp
                vis->setWorld(world_.get());
                vis->setWindowSize(1920, 1080);
                vis->setImguiSetupCallback(imguiSetupCallback);
                vis->setImguiRenderCallback(imguiRenderCallBack);
                vis->setKeyboardCallback(raisimKeyboardCallback);
                vis->setSetUpCallback(setupCallback);
                vis->setAntiAliasing(2);

                // starts visualizer thread
                vis->initApp();

                minicheetahVisual_ = vis->createGraphicalObject(minicheetah_, "MiniCheetah");
                if (flag_crucial) {
                    for (int i = 0; i < num_cube; i++) {
                        vis->createGraphicalObject(cubes[i], "cubes" + std::to_string(i), "yellow");
                    }
                }
                if (flag_terrain) {
                    topography_ = vis->createGraphicalObject(hm, "floor");
                } else {
                    topography_ = vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
                }

                desired_fps_ = 60.;
                vis->setDesiredFPS(desired_fps_);
            }
            e.seed(10);  // set the seed of random engine
            // std::cout<<"MINI_109"<<std::endl;
#if DEBUG0
            std::cout << "22" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
        }

        ~ENVIRONMENT() final = default;

        void init() final {

            // frame_max = ref.col(0).size();
            frame_max = ref.col(0).size() / 2;
            frame_len = int(max_time / control_dt_);
        }

        /*
         * randomly produce command and init the robot init state
         * according to the gait generator init robot init joint
         * return observation
         */
        void reset() final {
#if DEBUG0
            std::cout << "33" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            // reset the robot joint according to the gait pattern
            itera++;    // update environment reset time
            if (itera % 100000 == 0) std::cout << "env itera:" << itera << std::endl;
            current_time_ = (flag_manual) ? 0.0 : double(random()) / double(RAND_MAX);

            for (double &cmd : command_filtered) {
                cmd = 0;  // reset command, or the small velocity is impossible to be learned
            }

//            frame_idx = (frame_max - frame_len - 10) * double(random()) / double(RAND_MAX);
            if (flag_ManualTraj) {
                frame_idx = 0;
            } else {
                frame_idx = (frame_max - frame_len - 10) * sampling_reshape(double(random()) / double(RAND_MAX));
                frame_idx = (flag_manual) ? 0 : frame_idx;
            }

            torque_last.setZero(nJoints_);
            // frame_idx = 15000;
#if DEBUG0
            std::cout << "44" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            command_obs_update(true);       // randomly produce command at the very beginning
#if DEBUG0
            std::cout << "55" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            contact_obs_update(true);       // contact information update
#if DEBUG0
            std::cout << "66" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            for (int i = 0; i < 4; i++) {
                if (contact_[i] > 0.5) {
                    jointPgain.segment(6 + i * 3, 3) << abad_ratio * stiffness, stiffness, stiffness;
                } else {
                    jointPgain.segment(6 + i * 3, 3) << abad_ratio * stiffness_low, stiffness_low, stiffness_low;
                }
                minicheetah_->setPdGains(jointPgain, jointDgain);
            }

            // current_time_ = 0.0;
            float temp__ = 0.0f;
            // random the joint based the gait generator
//            for (int i = 0; i < nJoints_; i++) {
//                temp__ = rand() / float(RAND_MAX);
//                temp__ = (temp__ - 0.5f) / 0.5f * 0.3f + 1.0f;
//                random_init[7 + i] = temp__ * jointRef_[i];
//                pTarget12Last_[i] = temp__ * jointRef_[i];
//            }
            init_noise.setRandom(12);
            random_init.segment(7, 12) = jointRef_ * (init_noise * 0.3) + jointRef_;
            init_noise.setRandom(12);
            random_vel_init.tail(12) = jointDotRef_ * (init_noise * 0.3) + jointDotRef_;
            init_noise.setRandom(3);
            random_vel_init[0] = command_filtered[0] * (init_noise[0] * 0.2 + 1.0);
            random_vel_init[1] = command_filtered[1] * (init_noise[1] * 0.2 + 1.0);
            random_vel_init[5] = command_filtered[2] * (init_noise[2] * 0.2 + 1.0);
            // random init robot position
            if (not flag_manual) {
                // only when
                temp__ = rand() / float(RAND_MAX);
                random_init[0] = temp__ * 5.0 + (1.0 - temp__) * -5.0;
                temp__ = rand() / float(RAND_MAX);
                random_init[1] = temp__ * 5.0 + (1.0 - temp__) * -5.0;
//                Vec<3> euler;
//                Vec<4> quat;
//                temp__ = rand() / float(RAND_MAX);
//                euler[2] = temp__ * 2 * PI;
//                raisim::eulerVecToQuat(euler, quat);
//                random_init.segment(3, 4) << quat.e();  // rotation
            }
            // add rock attack
            if (flag_crucial) {
                flag_is_attack = false;
                meteoriteAttack(true, flag_is_attack);                 // reset the attack
            }

            // minicheetah_->setState(gc_init_, gv_init_);                              // fixed init
            // minicheetah_->setState(random_init, gv_init_);
            minicheetah_->setState(random_init, random_vel_init);
            updateObservation();     // update observation
            obDouble_last_ = obDouble_;
            contact_obs_update(false);  //  update contact state
            command_obs_update(false);    // create next time reference traj

            frame_idx++;            // update time
            current_time_ += control_dt_;
            // std::cout << "obs:" << obDouble_ << std::endl;
            if (visualizationCounter_)
                gui::rewardLogger.clean();
        }

        /*
         * receive action from controller and perform the action
         * according the
         */
        float step(const Eigen::Ref<EigenVec> &action) final {
            // ---------------------------------------------------------
            // transform neural network output to
            // action scaling
            pTarget12_ = action.cast<double>();
            pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
            pTarget12_ += actionMean_;
            pTarget12_ = (1.0 - filter_para) * pTarget12_ + filter_para * pTarget12Last_;
            action_noise.setRandom(12);
            pTarget12_ = pTarget12_ * (actionNoise * action_noise) + pTarget12_;
            pTarget_.tail(nJoints_) = pTarget12_;
            // pTarget_.tail(nJoints_) = gc_init_.tail(nJoints_);   // stand here
            // gait_generator();
            // pTarget_.tail(nJoints_) = jointRef_;
            pTarget12Last_ = pTarget12_;
            jointLast_ = gc_.tail(nJoints_);  // last time step joint angle
            // std::cout<<pTarget12Last_<<std::endl;
            torque = (pTarget12_ - gc_.tail(12)).cwiseProduct(jointPgain.tail(nJoints_)) -
                     (gv_.tail(12)).cwiseProduct(jointDgain.tail(12));
            if (!flag_MotorDynamics) {
                minicheetah_->setPdTarget(pTarget_, vTarget_);  // set the NN output as target position
                for (int i = 0; i < 4; i++) {
                    if (contact_[i] > 0.5) {
                        jointPgain.segment(6 + i * 3, 3) << abad_ratio * stiffness, stiffness, stiffness;
                    } else {
                        jointPgain.segment(6 + i * 3, 3) << abad_ratio * stiffness_low, stiffness_low, stiffness_low;
                    }
                    minicheetah_->setPdGains(jointPgain, jointDgain);
                }
            } else {
                minicheetah_->setPdGains(Eigen::VectorXd::Zero(gvDim_), Eigen::VectorXd::Zero(gvDim_));
                RealTorque(torque, gv_.tail(12), 12, true);  // recalculate torque
                torque_.tail(12) << torque;
                minicheetah_->setGeneralizedForce(torque_);
#if DEBUG2
                std::cout << "generailized force" << minicheetah_->getGeneralizedForce() << std::endl;
#endif
            }


            // ----------------------------------------------------------------------------------
            // pure torque control
            // torque_.tail(nJoints_) = pTarget12_;
            // minicheetah_->setGeneralizedForce(torque_);  // apply torque


            // gc_.head(7) = gc_init_.head(7);                 // hold body
            // gv_.head(6) = gv_init_.head(6);
            // minicheetah_->setState(gc_, gv_);

            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);  // update loop count
            auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);

            // -------------------------------------------
            // performance attack
            if (flag_crucial) {
                if (double (random()) / double(RAND_MAX) < AttackProbability){
                    meteoriteAttack(false, flag_is_attack);
                    flag_is_attack = true;
                }
                if (double (random()) / double(RAND_MAX) < 0.001){
                    // there is a very small probability to reset the attack
                    flag_is_attack = false;
                    meteoriteAttack(true, flag_is_attack);
                }
            }

            for (int i = 0; i < loopCount; i++) {
                world_->integrate();

                if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
                    raisim::OgreVis::get()->renderOneFrame();

                visualizationCounter_++;
            }

            updateObservation();                           // update observation expect joint ref
            DeepMimicReward_ = DeepMimicRewardUpdate();    // calculate reward before observation update
            // AttackProbability = DeepMimicReward_ *
            //                     DeepMimicReward_ *
            //                     DeepMimicReward_ *
            //                     DeepMimicReward_;          // according to the reward determine attack probability
            AttackProbability = 1.0 / (double) frame_len;     // make sure only be attack once
            command_obs_update(false);            // update joint reference after reward is calculated
            contact_obs_update(false);
            current_time_ += control_dt_;                   // update current time
            frame_idx += 1;
            if (visualizeThisStep_) {
                auto vis = raisim::OgreVis::get();
                // vis->select(minicheetahVisual_->at(0), false);
                // vis->select(topography_->at(0), false);
                if (flag_fix_camera_to_ground) {
                    vis->select(topography_->at(0), false);
                } else {
                    vis->select(minicheetahVisual_->at(0), false);
                }
                if (not flag_manual) {
                    vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.3), 3, true);
                    // vis->getCameraMan()->setYawPitchDist(Ogre::Radian(-1.57), Ogre::Radian(-1.3), 3, true);
                }
            }
#if DEBUG1
            std::cout << "88" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            return DeepMimicReward_;
        }

        /*
         * crucial learning
         * add some blocks and balls to attack the robot, increase distribution
         * */
        void meteoriteAttack(bool flag_reset, bool is_attack) {
            // std::cout<<"I'm here"<<std::endl;
            if (flag_reset) {
                for (int i = 0; i < num_cube; i++) {
                    cubes[i]->setPosition(circle_place(cube_place_radius, i, num_cube, 2.0));
                    cubes[i]->setBodyType(raisim::BodyType::STATIC);
                }
            } else {
                if (not is_attack) {
                    Eigen::Vector3d vel(0, 0, 0);
                    Eigen::Vector3d omega(0, 0, 0);
                    for (int i = 0; i < num_cube; i++) {
                        cubes[i]->setBodyType(raisim::BodyType::DYNAMIC);
                        vel = minicheetah_->getBasePosition().e() - cubes[i]->getPosition(); // attack towards robot
                        vel << 5 * vel[0] * (1.0 + (*noise_attack)(e)),
                                5 * vel[1] * (1.0 + (*noise_attack)(e)),
                                5 * vel[2] * (1.0 + (*noise_attack)(e));
                        cubes[i]->setVelocity(vel, omega);
                    }
                }
            }
        }

        void updateExtraInfo() final {
            extraInfo_["EndEffectorReward(0.15)"] = EndEffectorReward;
            extraInfo_["Height_Keep_Reward(0.1)"] = BodyCenterReward;
            extraInfo_["base height"] = gc_[2];
            extraInfo_["Balance_Keep_Reward(0.1)"] = BodyAttitudeReward;
            extraInfo_["JointReward(0.65)"] = JointReward;
            extraInfo_["VelocityReward(0.2)"] = VelocityReward;
        }

        /*
         * this method is used to update observation expect reference joint
         * height X1, axis_z X3, joint X12, vel X3, omega X3, joint_dot X12, contact X4
         */
        void updateObservation() {
            minicheetah_->getState(gc_, gv_);  // get dynamics state of robot

            obDouble_.setZero(obDim_);       // clear observation buffer
            obScaled_.setZero(obDim_);

            // phase update
            if (flag_manual or flag_ManualTraj) {
                obDouble_[3] = sin(2 * PI * current_time_ / period_);
                obDouble_[4] = cos(2 * PI * current_time_ / period_);
                // std::cout<<"current time: "<<current_time_<<" phase: "<<obDouble_[3]<<" "<<obDouble_[4]<<std::endl;
            } else {
                obDouble_.segment(3, 2) << ref.row(frame_idx).transpose().segment(25, 2).cast<double>();
            }
            // obDouble_[4] = cos(2 * PI * current_time_ / period_);
            // joint angle update
            Eigen::VectorXd joint_noise;
            joint_noise.setRandom(12);
            // obDouble_.segment(5, 12) = gc_.tail(12) * (joint_noise * jointNoise * noise_flag) + gc_.tail(12);
            obDouble_.segment(5, 12) = (joint_noise * jointNoise * noise_flag) + gc_.tail(12);

            // joint angle velocity update
            joint_noise.setRandom(12);
            obDouble_.segment(17, 12) = (joint_noise * jointVelocityNoise * noise_flag) + gv_.tail(12);

            // posture update
            raisim::Mat<3, 3> rot;
            raisim::Vec<4> quat;
            quat[0] = gc_[3];
            quat[1] = gc_[4];
            quat[2] = gc_[5];
            quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot);
            bodyFrameMatrix_ = rot;
            obDouble_.segment(29, 3) << rot.e().row(2)(0) + (*noise_posture)(e) * noise_flag,
                    rot.e().row(2)(1) + (*noise_posture)(e) * noise_flag,
                    rot.e().row(2)(2) + (*noise_posture)(e) * noise_flag;

            // transform from the global coordinate
            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
            obDouble_.segment(32, 3) << bodyAngularVel_(0) + noise_flag * (*noise_omega)(e),
                    bodyAngularVel_(1) + noise_flag * (*noise_omega)(e),
                    bodyAngularVel_(2) + noise_flag * (*noise_omega)(e);

        }

        /*
         * this function is used to update command observation
         * ramdomly produce command, which is used for gait generation to produce reference joint
         */
        void command_obs_update(bool flag_reset) {

            if (flag_manual) {
                // since now, do nothing
                // in manual control mode, this should be change in controller scripts
                // for AI controller, user can just modify nn input layer directly
                ;
#if DEBUG1
                command_filtered[0] = 1.0;
#endif

            } else {
#if DEBUG0
                std::cout << "3333" << std::endl;
                std::cout << ref.row(0) << std::endl;
#endif
                // obDouble_.head(3) << ref.row(frame_idx).transpose().tail(3).cast<double>();
                if (flag_ManualTraj) {
                    float temp__ = 0.0;
                    temp__ = rand() / float(RAND_MAX);
                    if (temp__ < 2.0 / (max_time / control_dt_) or flag_reset) {
                        // randomly modify control command
                        // since this is uniform random distribution, the command update almost T/dt/100 times
                        // update command
                        temp__ = rand() / float(RAND_MAX);
                        if (temp__ < 0.2) {
                            for (auto cmd : command) {
                                cmd = 0.0;
                            }
                        }
                        if (0.2 < temp__ and temp__ <= 1.0) {
                            temp__ = rand() / float(RAND_MAX);
                            command[0] = temp__ * Vx_max + (1.0 - temp__) * Vx_min;
                        } else {
                            if (0.5 < temp__ and temp__ <= 0.75) {
                                temp__ = rand() / float(RAND_MAX);
                                command[1] = temp__ * Vy_max + (1.0 - temp__) * Vy_min;
                            } else {
                                temp__ = rand() / float(RAND_MAX);
                                command[2] = temp__ * omega_max + (1.0 - temp__) * omega_min;
                            }
                        }
                    }
                    if (flag_reset) {
                        command_filtered[0] = command[0];
                        command_filtered[1] = command[1];
                        command_filtered[2] = command[2];
                    } else {
                        for (int i = 0; i < 3; i++) {
                            command_filtered[i] =
                                    command_filtered[i] * cmd_update_param + command[i] * (1 - cmd_update_param);
                        }
                    }

                    obDouble_.head(3) << command_filtered[0],
                            command_filtered[1],
                            command_filtered[2];
                    gait_generator_manual(flag_reset);
                } else {
                    obDouble_.head(3) << ref.row(frame_idx).transpose().segment(27, 3).cast<double>();
                    command_filtered[0] = obDouble_.data()[0];
                    command_filtered[1] = obDouble_.data()[1];
                    command_filtered[2] = obDouble_.data()[2];
                    gait_generator();
                }

#if DEBUG0
                std::cout << "333333" << std::endl;
                // std::cout << ref->row(0) << std::endl;
#endif
//                obDouble_.head(3) << command_filtered[0],
//                        command_filtered[1],
//                        command_filtered[2];
            }
//            if (flag_ManualTraj) {
//                gait_generator_manual(flag_reset);
//            } else {
//                gait_generator();
//            }
#if DEBUG1
            std::cout << "ref joint: " << jointRef_.transpose() << std::endl;
            std::cout << "ref joint_dot: " << jointDotRef_.transpose() << std::endl;
#endif

        }

        /*
         * contact_obs_update is used to check whether the leg contact with ground
         * since now, if the hoof contact with ground, the value will close to 1
         * using the contact force and shape function to produce the contact signal
         */
        void contact_obs_update(bool flag_reset) {
            // note: contact is setted to zero during every update observation
            // shape function: obs = 1 - exp(alpha*contact_force)
            // std::cout << "contact:";
            if (not flag_TimeBasedContact) {
                for (double &cc : contact_) {
                    cc = 0.0;
                }
                for (auto &contact: minicheetah_->getContacts()) {
                    // minicheetah_->printOutBodyNamesInOrder();
                    if (minicheetah_->getBodyIdx("shank_fr") == contact.getlocalBodyIndex()) {
                        // obDouble_(34) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "fr:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        contact_[0] = 1.0;
                    } else if (minicheetah_->getBodyIdx("shank_fl") == contact.getlocalBodyIndex()) {
                        // obDouble_(35) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "fl:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        contact_[1] = 1.0;
                    } else if (minicheetah_->getBodyIdx("shank_hr") == contact.getlocalBodyIndex()) {
                        // obDouble_(36) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "hr:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        contact_[2] = 1.0;
                    } else if (minicheetah_->getBodyIdx("shank_hl") == contact.getlocalBodyIndex()) {
                        // obDouble_(37) = 1 - exp(contact_para * contact.getImpulse()->e().squaredNorm());
                        // std::cout << "hl:"<<contact_para * contact.getImpulse()->e().squaredNorm();
                        contact_[3] = 1.0;
                    } else { ;  // amazing saturation
                    }
                }
                for (int i = 0; i < 4; i++) {
//                    contact_filtered[i] = (flag_reset) ? contact_[i] : (contact_filtered[i] * filter_para +
//                                                                        (1.0 - filter_para) * contact_[i]);
                    contact_filtered[i] = contact_[i];
                    // obDouble_(34 + i) = contact_filtered[i];
                }
                // std::cout << obDouble_.segment(34, 4).transpose() << std::endl;
            } else {
                // time based contact
                float real_phase;
                float temp;
                for (double &cc : contact_) {
                    cc = 0.0;
                }
                for (int i = 0; i < 4; i++) {
                    real_phase = current_time_ + phase_[i] * period_;
                    real_phase = fmod(real_phase, period_) / period_;
                    // temp = (real_phase < lam_) ? 2.0 * real_phase : 0.0;
                    // contact_filtered[i] = ((temp < 1.0) and (temp > 0.01)) ? 1.0 : 0.0;
                    // obDouble_(34 + i) = contact_filtered[i];
                    contact_filtered[i] = (real_phase < lam_) ? 1.0 : 0.0;
                    contact_[i] = contact_filtered[i];
                    // obDouble_(34 + i) = contact_filtered[i];
                }
                // obDouble_[34] = cos(2 * PI * current_time_ / period_);
                // obDouble_[35] = sin(2 * PI * current_time_ / period_);
                // obDouble_[36] = sin(4 * PI * current_time_ / period_);
                // obDouble_[37] = sin(6 * PI * current_time_ / period_);
            }

        }

        /*
         * scale the observation and return to neural network
         */
        void observe(Eigen::Ref<EigenVec> ob) final {
            // convert it to float
            if (flag_ObsFilter) {
                obDouble_.tail(obDim_ - 5) << obDouble_.tail(obDim_ - 5) * ObsFilterAlpha +
                                              obDouble_last_.tail(obDim_ - 5) * (1.0 - ObsFilterAlpha);
                obDouble_last_ = obDouble_;
            } else { ;
            }
            obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_);  // (obDouble_-obMean_)./obStd_
            ob = obScaled_.cast<float>();
            // std::cout << "-----------------------------" << std::endl;
            // std::cout << "ob_double" << obDouble_.transpose() << std::endl;
            // std::cout << "ob_scale" << obScaled_.transpose() << std::endl;
            // std::cout << "*****************************" << std::endl;
            // std::cout << "raisim_ob_scaled:" << obScaled_.segment(16, 6).transpose() << std::endl;
        }

        /*
         * return all original state
         */
        void OriginState(Eigen::Ref<EigenVec> origin_state) {
            Eigen::VectorXd ss;
            //ss.setZero(gcDim_ + gvDim_);
            //ss << gc_, gv_;
            ss.setZero(gcDim_ + gvDim_ + 4);
            ss << gc_, gv_, contact_filtered[0], contact_filtered[1], contact_filtered[2], contact_filtered[3];
            origin_state = ss.cast<float>();
        }

        /*
         * return origin state dim
         */
        int GetOriginStateDim() {
            //return (gcDim_ + gvDim_);
            return (gcDim_ + gvDim_ + 4);
        }

        /*
         * according to gait pattern generator & current state update rewards
         * the form of the cost function mainly refer to https://github.com/xbpeng/DeepMimic
         * for more detail, please click this link
         * return DeepMimic reward (double)
         */
        double DeepMimicRewardUpdate() {
            // =================================================================
            // <<<<<<<<<<<<Calculate end effector reward>>>>>>>>>>>>>>>>>>>>>>>>
            bodyPos_ << gc_[0], gc_[1], gc_[2];
            Eigen::Vector3d temp;
            for (int i = 0; i < 4; i++) {
                minicheetah_->getFramePosition(minicheetah_->getFrameIdxByName(ToeName_[i]), TempPositionHolder_);
                // here, the hoof's radius is ignored, this may be wrong, please check it
                // make sure to check it !!!!!!
                temp << TempPositionHolder_[0], TempPositionHolder_[1], TempPositionHolder_[2];
                EndEffector_.segment(3 * i, 3) << bodyFrameMatrix_.e().transpose() * (temp - bodyPos_);
            }
            // std::cout<<BOLDYELLOW<<EndEffector_.transpose()<<std::endl;
            EndEffectorReward = (EndEffector_ - EndEffectorRef_).squaredNorm();
            EndEffectorReward = EECoeff * exp(-40 * EndEffectorReward);

            // ==================================================================
            // <<<<<<<<<<<<<<<<< Calculate body center error >>>>>>>>>>>>>>>>>>>>
            // the designd trajectory along the x direction, designed speed is gait_step / period
            // bodyPos_(0) = bodyPos_(0) + gait_step_ / period_ * current_time_;
            // std::cout<<BOLDYELLOW<<bodyPos_<<std::endl;
            bodyPos_(0) = 0;
            bodyPos_(1) = 0;
            // std::cout<<BOLDCYAN<<bodyPos_<<std::endl;

            // temp << 0, 0, ref.row(0)[24];
            // temp << 0, 0, ref.row(frame_idx).transpose().segment(24, 1)[0];
            temp << 0, 0, stand_height_;
            // BodyCenterReward = bodyPos_.segment(1, 1).squaredNorm();  // here is a problem, z axis should be a desired constant
            BodyCenterReward = (bodyPos_ - temp).squaredNorm();
            BodyCenterReward = BodyPosCoeff * exp(-80 * BodyCenterReward);

            // posture reward
            // double BodyAttitudeReward = (gv_.segment(3,4) - ForwardDirection_.tail(4)).squaredNorm();  // keep forward & balance
            // temp << ref.row(frame_idx).transpose().segment(30, 3).cast<double>();
            BodyAttitudeReward = obDouble_.segment(29, 2).squaredNorm();  // make the sobot keep balance
            BodyAttitudeReward = (obDouble_.segment(29, 3) - temp).squaredNorm();
            BodyAttitudeReward = BodyAttiCoeff * exp(-80 * BodyAttitudeReward);

            // DirectionKeepReward = obDouble_.segment(22, 3).squaredNorm();
            // temp << 1, 0, 0;
            // DirectionKeepReward = (obDouble_.tail(3) - temp).squaredNorm();
            // DirectionKeepReward = 0.0 * exp(-2 * DirectionKeepReward);

            // ==================================================================
            // <<<<<<<<<<<<<<<<<< Calculate Joint Mimic Reward>>>>>>>>>>>>>>>>>>>
            JointReward = (jointRef_ - gc_.tail(12)).squaredNorm();
            JointReward = JointMimicCoeff * 0.25 * exp(-2.0 * JointReward);
            JointDotReward = (jointDotRef_ - gv_.tail(12)).squaredNorm();
            JointDotReward = JointMimicCoeff * 0.75 * exp(-control_dt_ * JointDotReward);


            //===================================================================
            // <<<<<<<<<<<<<<<<<<<<<<<<< Velocity Reward >>>>>>>>>>>>>>>>>>>>>>>>
            // this reward is used to follow user command
            bodyLinearVelRef_ << command_filtered[0], command_filtered[1], 0;
            bodyAngularVelRef_ << 0.0, 0.0, command_filtered[2];
            VelocityReward = VelKeepCoeff / 2 * exp(-2 * (bodyLinearVel_ - bodyLinearVelRef_).squaredNorm()) +
                             VelKeepCoeff / 2 * exp(-2 * (bodyAngularVel_ - bodyAngularVelRef_).squaredNorm());

            // ==================================================================
            // <<<<<<<<<<<<<<<<<<<<<<<< Torque Reward >>>>>>>>>>>>>>>>>>>>>>>>>>>
//            torque = (pTarget12_ - gc_.tail(12)).cwiseProduct(jointPgain.tail(nJoints_)) -
//                     (gv_.tail(12)).cwiseProduct(jointDgain.tail(12));
            torque = torque.cwiseQuotient(torque_limit);
            // TorqueReward = TorqueCoeff * exp(-1.0 * torque.squaredNorm());
            TorqueReward = TorqueCoeff / 2.0 * exp(-1.0 * torque.squaredNorm()) +
                           TorqueCoeff / 2.0 * exp(-1.0 / control_dt_ * (torque - torque_last).squaredNorm());
            torque_last = torque;
            // update gui log
            if (visualizeThisStep_) {
                gui::rewardLogger.log("EndEffector", EndEffectorReward);
                gui::rewardLogger.log("HeightKeep", BodyCenterReward);
                gui::rewardLogger.log("BalanceKeep", BodyAttitudeReward);
                gui::rewardLogger.log("Joint", JointReward);
                gui::rewardLogger.log("JointDot", JointDotReward);
                gui::rewardLogger.log("Velocity", VelocityReward);
                gui::rewardLogger.log("Torque", TorqueReward);
                gui::rewardLogger.log("Forward", command_filtered[0]);
                gui::rewardLogger.log("lateral", command_filtered[1]);
                gui::rewardLogger.log("yaw", command_filtered[2]);
            }
            return (EndEffectorReward + BodyCenterReward + JointReward + JointDotReward +
                    VelocityReward + BodyAttitudeReward + TorqueReward);
        }

        /*
         * detect whether robot is under terrible situation and terminate this explore
         * */
        bool isTerminalState(float &terminalReward) final {
            // used to detect whether to terminate the episode
            terminalReward = float(terminalRewardCoeff_);
//            std::cout<<obDouble_.segment(0,4).transpose()<<std::endl;
            // if the height of the robot is not in 0.28~0.39 or The body is too inclined
            // if(obDouble_[0]<0.28 or obDouble_[0]>0.39 or obDouble_[3]<0.7){
            if (gc_[2] < 0.20 or gc_[2] > 0.50 or obDouble_[31] < 0.7) {
//                if(obDouble_[0] < 0.25){
//                    std::cout<<"body too low"<<std::endl;
//                }
//                if(obDouble_[0] > 0.4){
//                    std::cout<<"body too high"<<std::endl;
//                }
//                if(obDouble_[3]<0.7){
//                    std::cout<<"body too declining"<<std::endl;
//                }
                return true;
            } else {
                terminalReward = 0.f;
                return false;
            }
        }

        void setSeed(int seed) final {
            std::srand(seed);
        }

        void close() final {}

        /*
         * this function is used to load all parameter from the yaml file
         * switch the environment to be manual mode or auto mode
         * switch the ground to be plane or terrain
         * load gait parameter
         * something else
         */
        void parameter_load_from_yaml(const YAML::Node &cfg) {
            // ------------------------------------------------------------
            // load gait parameter
            READ_YAML(double, abad_, cfg["abad"]);
            READ_YAML(double, period_, cfg["period"]);
            READ_YAML(double, lam_, cfg["lam"]);
            READ_YAML(double, stand_height_, cfg["stand_height"]);
            READ_YAML(double, up_height_, cfg["up_height"]);
            up_height_max_ = up_height_;
            READ_YAML(double, down_height_, cfg["down_height"]);
            READ_YAML(double, gait_step_, cfg["gait_step"]);
            READ_YAML(double, Vx_max, cfg["Vx"]);
            READ_YAML(double, Vy_max, cfg["Vy"]);
            Vy_min = -Vy_max;
            READ_YAML(double, omega_max, cfg["Omega"]);
            omega_min = -omega_max;
            READ_YAML(double, Lean_middle_front, cfg["LeanFront"]);
            READ_YAML(double, Lean_middle_hind, cfg["LeanHind"]);
            // ------------------------------------------------------------
            // load mode parameter
            READ_YAML(bool, flag_terrain, cfg["Terrain"]);
            READ_YAML(bool, flag_manual, cfg["Manual"]);
            READ_YAML(bool, flag_crucial, cfg["Crutial"]);
            READ_YAML(bool, flag_filter, cfg["Filter"]);
            READ_YAML(bool, flag_fix_camera_to_ground, cfg["Camera"]);
            READ_YAML(bool, flag_StochasticDynamics, cfg["StochasticDynamics"]);
            READ_YAML(bool, flag_HeightVariable, cfg["HeightVariable"]);
            READ_YAML(bool, flag_TimeBasedContact, cfg["TimeBasedContact"]);
            READ_YAML(bool, flag_ManualTraj, cfg["ManualTraj"]);
            READ_YAML(bool, flag_MotorDynamics, cfg["MotorDynamics"]);
            READ_YAML(bool, flag_ObsFilter, cfg["ObsFilter"]);
            // ------------------------------------------------------------
            // load reward parameter
            READ_YAML(double, terminalRewardCoeff_, cfg["terminalRewardCoeff"]);
            READ_YAML(double, EECoeff, cfg["EndEffectorRewardCoeff"]);
            READ_YAML(double, BodyPosCoeff, cfg["BodyPosRewardCoeff"]);
            READ_YAML(double, BodyAttiCoeff, cfg["BodyAttitudeRewardCoeff"]);
            READ_YAML(double, JointMimicCoeff, cfg["JointRewardCoeff"]);
            READ_YAML(double, VelKeepCoeff, cfg["VelRewardCoeff"]);
            READ_YAML(double, TorqueCoeff, cfg["TorqueCoeff"]);
            READ_YAML(double, actionNoise, cfg["ActionNoise"]);
            READ_YAML(double, noise_flag, cfg["ObsNoise"]);
            // ------------------------------------------------------------
            // load control parameter
            READ_YAML(double, stiffness, cfg["Stiffness"]);
            READ_YAML(double, stiffness_low, cfg["Stiffness_Low"]);
            READ_YAML(double, abad_ratio, cfg["AbadRatio"]);
            READ_YAML(double, damping, cfg["Damping"]);
            READ_YAML(double, freq, cfg["Freq"]);
            READ_YAML(double, max_time, cfg["max_time"]);
            READ_YAML(int, num_cube, cfg["CubeNum"]);
        }

        /*
         * generate reference joint angle
         * */
        void gait_generator() {
#if DEBUG0
            std::cout << "4444" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
            jointRef_ << ref.row(frame_idx).transpose().segment(0, 12).cast<double>();
            jointDotRef_ << ref.row(frame_idx).transpose().segment(12, 12).cast<double>();
#if DEBUG1
            std::cout << "frame_id: " << frame_idx << std::endl;
            std::cout << "ref_traj: " << jointRef_.transpose() << std::endl;
            std::cout << "col: " << ref.col(0).segment(0, 12) << std::endl;
            std::cout << "row: " << ref.row(frame_idx).segment(0, 12).cast<double>() << std::endl;
#endif
#if DEBUG0
            std::cout << "444444" << std::endl;
            // std::cout << ref->row(0) << std::endl;
#endif
        }

        /*
         * 3D inverse kinematics solver
         */
        void inverse_kinematics(double x, double y, double z,
                                double l_hip, double l_thigh, double l_calf,
                                double *theta, bool is_right) {

            if (sqrt(x * x + y * y + z * z) > max_len) {
                x = x / max_len;
                y = y / max_len;
                z = z / max_len;
            }
            double temp, temp1, temp2 = 0.0;
            if (is_right) {
                temp = (-z * l_hip - sqrt(y * y * (z * z + y * y - l_hip * l_hip))) / (z * z + y * y);
                if (abs(temp) <= 1) {
                    theta[0] = asin(temp);
                } else {
                    std::cout << "error1" << std::endl;
                }
            } else {
                temp = (z * l_hip + sqrt(y * y * (z * z + y * y - l_hip * l_hip))) / (z * z + y * y);
                if (abs(temp) <= 1) {
                    theta[0] = asin(temp);
                } else {
                    std::cout << "error1" << std::endl;
                }
            }
            double lr = sqrt(x * x + y * y + z * z - l_hip * l_hip);
            lr = (lr > (l_thigh + l_calf)) ? (l_thigh + l_calf) : lr;
            temp = (l_thigh * l_thigh + l_calf * l_calf - lr * lr) / 2 / l_thigh / l_calf + 1e-5;
            if (abs(temp) <= 1) {
                theta[2] = -(PI - acos(temp));
            } else {
                std::cout << "error2" << std::endl;
                std::cout << "temp" << temp << std::endl;
                std::cout << "lr=" << lr << std::endl;
            }
            temp1 = x / sqrt(y * y + z * z) - 1e-10;
            temp2 = (lr * lr + l_thigh * l_thigh - l_calf * l_calf) / 2 / lr / l_thigh - 1e-5;
            if (abs(temp1) <= 1 and abs(temp2) <= 1) {
                theta[1] = acos(temp2) - asin(temp1);
            } else {
                std::cout << "error3" << std::endl;
                std::cout << "abs(temp1)" << abs(temp1) << std::endl;
                std::cout << "abs(temp2)" << abs(temp2) << std::endl;
                std::cout << "lr=" << lr << std::endl;
            }
        }


        /*
         * manual traj gait generator
         */
        void gait_generator_manual(bool is_first) {
            // temp variable
            double real_phase = 0.0;
            double toe_x = 0.0;
            double toe_y = 0.0;
            double toe_z = 0.0;

            double l1 = l_thigh_;
            double l2 = l_calf_;
            double temp[3] = {0.0, 0.0, 0.0};

            double anti_flag = 1.0;
            EndEffectorRef_.setZero(nJoints_);

            //according to the command to update the gait parameter
            gait_step_ = command_filtered[0] / 2.0 * period_;        // command speed along x * period
            side_step_ = command_filtered[1] / 2.0 * period_;        // command speed along y * period
            // rot_step_ = command_filtered[2] * period_ * 0.22;  // command rotation speed * radius * period
            rot_step_ = command_filtered[2] * period_ * 0.4;
            //change the maximum height of the hoof from the ground
            if (flag_HeightVariable) {
                double ratio = 0.0;
                ratio = std::abs(command_filtered[0]) / Vx_max;
                if (Vy_max > 0) {
                    ratio = fmax(ratio, std::abs(command_filtered[1]) / Vy_max);
                }
                if (omega_max > 0) {
                    ratio = fmax(ratio, std::abs(command_filtered[2] / omega_max));
                }
                up_height_ = (ratio > 0.1) ? up_height_max_ : ratio * up_height_max_;
            }
            Eigen::Vector3d p0, pf, toe;
            double temp_offset[4];
            temp_offset[0] = -l_hip_ + Lean_middle_front;
            temp_offset[1] = l_hip_ - Lean_middle_front;
            temp_offset[2] = -l_hip_ + Lean_middle_hind;
            temp_offset[3] = l_hip_ - Lean_middle_hind;
            if (is_first) {
                // calculate last step to get the reference joint velocity
                for (int i = 0; i < 4; i++) {
                    real_phase =
                            current_time_ + phase_[i] * period_ - control_dt_;  // calculate last time joint reference
                    real_phase = fmod(real_phase, period_) / period_;
                    anti_flag = (i < 2) ? 1.0 : -1.0;
                    // calculate the toe position relative to the hip
                    double temp_r = 0;
                    if (real_phase < lam_) {
                        temp_r = real_phase / lam_;
                        p0 << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                        pf << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                        toe = cubicBezier(p0, pf, temp_r);

                    } else {
                        temp_r = (real_phase - lam_) / (1.0 - lam_);
//                        if (temp_r < 0.5) {
//                            p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
//                            pf << 0.0, 0.0, up_height_ - stand_height_;
//                            toe = cubicBezier(p0, pf, temp_r / 0.5);
//                        } else {
//                            p0 << 0.0, 0.0, up_height_ - stand_height_;
//                            pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
//                            toe = cubicBezier(p0, pf, (temp_r - 0.5) / 0.5);
//                        }
                        pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                        p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                        toe = Bezier2(p0, pf, temp_r, up_height_);
                    }
                    toe_x = toe(0);
                    toe_y = toe(1);
                    toe_z = toe(2);
                    inverse_kinematics(toe_x, toe_y + temp_offset[i], toe_z,
                                       l_hip_, l_thigh_, l_calf_, temp,
                                       i == 0 or i == 2);
                    jointRefLast_[3 * i + 0] = temp[0];
                    jointRefLast_[3 * i + 1] = -temp[1];
                    jointRefLast_[3 * i + 2] = -temp[2];
                }
            }
            for (int i = 0; i < 4; i++) {
                real_phase = current_time_ + phase_[i] * period_;
                real_phase = fmod(real_phase, period_) / period_;
                anti_flag = (i < 2) ? 1.0 : -1.0;
                // calculate the toe position relative to the hip
                double temp_r = 0;
                if (real_phase < lam_) {
                    temp_r = real_phase / lam_;
                    p0 << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                    pf << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                    toe = cubicBezier(p0, pf, temp_r);

                } else {
                    temp_r = (real_phase - lam_) / (1.0 - lam_);
//                    if (temp_r < 0.5) {
//                        p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
//                        pf << 0.0, 0.0, up_height_ - stand_height_;
//                        toe = cubicBezier(p0, pf, temp_r / 0.5);
//                    } else {
//                        p0 << 0.0, 0.0, up_height_ - stand_height_;
//                        pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
//                        toe = cubicBezier(p0, pf, (temp_r - 0.5) / 0.5);
//                    }
                    pf << gait_step_ / 2.0, side_step_ / 2.0 + anti_flag * rot_step_ / 2.0, -stand_height_;
                    p0 << -gait_step_ / 2.0, -side_step_ / 2.0 + -anti_flag * rot_step_ / 2.0, -stand_height_;
                    toe = Bezier2(p0, pf, temp_r, up_height_);
                }
                toe_x = toe(0);
                toe_y = toe(1);
                toe_z = toe(2);
                inverse_kinematics(toe_x, toe_y + temp_offset[i], toe_z, l_hip_, l_thigh_, l_calf_, temp,
                                   i == 0 or i == 2);
                jointRef_[3 * i + 0] = temp[0];
                jointRef_[3 * i + 1] = -temp[1];
                jointRef_[3 * i + 2] = -temp[2];
                EndEffectorRef_[3 * i + 0] = toe_x;
                EndEffectorRef_[3 * i + 1] = toe_y;
                EndEffectorRef_[3 * i + 2] = toe_z;
            }
            jointDotRef_ = (jointRef_ - jointRefLast_) / control_dt_;
            jointRefLast_ = jointRef_;

            EndEffectorRef_ = EndEffectorRef_ + EndEffectorOffset_;
        }

        /*
         * set reference trajectory database
         */
        void set_ref(Eigen::MatrixXf pr) {
            // ref = std::move(pr);
            ref = pr;
        }

        //-----------------------------------------
        // static Eigen::MatrixXf ref;


    private:

        // visualization related definition
        bool visualizable_ = false;
        std::normal_distribution<double> distribution_;
        raisim::ArticulatedSystem *minicheetah_;
        std::vector<GraphicObject> *minicheetahVisual_;
        std::vector<GraphicObject> *topography_;

        double desired_fps_ = 60.;
        int visualizationCounter_ = 0;

        // policy & robot related definition
        int gcDim_, gvDim_, nJoints_;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_, torque_;
        Eigen::VectorXd random_init;
        Eigen::VectorXd random_vel_init;
        Eigen::VectorXd pTarget12Last_;
        Eigen::VectorXd jointLast_;  // store last time joint angle
        Eigen::VectorXd jointPgain, jointDgain;
        Eigen::VectorXd torque;         // estimate joint torque of PD controller
        Eigen::VectorXd torque_limit;  // the boundary of torque
        Eigen::VectorXd torque_last;  // store the last time torque, try to smooth the output

        // lost function related definition
        double terminalRewardCoeff_ = -10.;

        Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
        Eigen::VectorXd obDouble_, obScaled_, obDouble_last_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        Eigen::Vector3d bodyLinearVelRef_, bodyAngularVelRef_;

        // gait pattern parameter define, those parameter can be change according to yaml configuration file
        Eigen::VectorXd phase_;
        double current_time_ = 0.0;   // current time
        double abad_ = 0.0;            // control the reference abad joint
        double period_ = 0.5;         // default gait period
        double lam_ = 0.65;           // proportion of time the robot touch the ground in each cycle
        double stand_height_ = 0.30;
        double up_height_ = 0.02;     // height of toe raise
        double up_height_max_ = 0.02;  // record the maximum height of toe raise
        double down_height_ = 0.02;   // height of toe embedded in the ground
        double gait_step_ = 0.09;     // length of one gait
        double side_step_ = 0.0;      // control the speed along the y axis
        double rot_step_ = 0.0;      // control the rotation speed
        double l_thigh_ = 0.209;      // length of the thigh
        double l_calf_ = 0.2175;        // length of the calf    # 0.19+0.0275
        // double l_calf_ = 0.19;        // ignore hoof size
        double l_hip_ = 0.085;         // length of hip along y direction
        double max_len = 0.0;
        double abad_ratio = 1.0;      // scaling factor, for abad joint stiffness and damping
        Eigen::VectorXd jointRef_;    // store the reference joint angle at current time, length should be 12
        Eigen::VectorXd jointRefLast_;
        Eigen::VectorXd jointDotRef_;  // store the reference joint angle at last time, length should be 12

        raisim::Vec<3> TempPositionHolder_;  // tempory position holder to get toe position
        Eigen::VectorXd EndEffector_;        // storage minicheetah storage four toe position relative to body frame
        Eigen::VectorXd EndEffectorRef_;     // reference end effector traj updated by pattern_generator(), relative to body frame
        Eigen::VectorXd EndEffectorOffset_;  // offset between hip coordinates and body center
        std::string ToeName_[4] = {"toe_fr_joint", "toe_fl_joint", "toe_hr_joint",
                                   "toe_hl_joint"};  // storage minicheetah toe frame name
        Eigen::Vector3d bodyPos_;
        raisim::Mat<3, 3> bodyFrameMatrix_;
        double DeepMimicReward_ = 0;
        Eigen::VectorXd ForwardDirection_;

        std::vector<raisim::Box *> cubes;  // for crucial learning, box will attack the robot
        int num_cube = 10;                 // cube number
        double cube_len = 0.08;            // length of cube
        double cube_mass = 0.4;           // default mass of cube
        double cube_place_radius = 3.0;    // cube place radius

        double EndEffectorReward = 0;
        double BodyCenterReward = 0;
        double BodyAttitudeReward = 0;
        double JointReward = 0;
        double JointDotReward = 0;
        double VelocityReward = 0;
        double TorqueReward = 0;

        double actionNoise = 0.1;
        double jointNoise = 0.002;//0.002;
        double jointVelocityNoise = 0.8;//0.8;
        // double observer_noise_amplitude = 1.1;

        double noise_flag = 1.0;
        default_random_engine e;
        // double noise_std_z = 0.01  // default height estimate noise standard deviation
        // normal_distribution<float>* noise_z;
        normal_distribution<float> *noise_z = new normal_distribution<float>(0.0, 0.005);
        normal_distribution<float> *noise_posture = new normal_distribution<float>(0.0, 0.02);// 0.02);
//        normal_distribution<float> *noise_posture = new normal_distribution<float>(0.0, 0.0);
        normal_distribution<float> *noise_omega = new normal_distribution<float>(0.0, 0.5);// 0.5);
//        normal_distribution<float> *noise_omega = new normal_distribution<float>(0.0, 0.0);

        normal_distribution<float> *noise_vel = new normal_distribution<float>(0.0, 0.5);
        normal_distribution<float> *noise_joint_vel = new normal_distribution<float>(0.0, 0.8);
        normal_distribution<float> *noise_attack = new normal_distribution<float>(0.0, 0.15);
        Eigen::VectorXd action_noise;
        Eigen::VectorXd init_noise;

        double filter_para = 0.0;    // action = filter_para * action_now + (1-filter_para) * action_last

        bool flag_manual = false;    // control robot automatically or manually
        bool flag_terrain = false;   // terrain ground or flat
        bool flag_crucial = false;   // whether add the difficulty of learning
        bool flag_filter = false;    // whether to add action filter
        bool flag_is_attack = false;  // flag to show whether the robot is under attack
        bool flag_fix_camera_to_ground = false;  // flag to control the camera view point, fixed to ground or robot
        bool flag_StochasticDynamics = false;    // flag to control whether disturb robot's dynamics
        bool flag_HeightVariable = false;        // flag to control whether the height of the foot lift is variable
        bool flag_TimeBasedContact = false;      // flag to control whether use the real contact or time based contact
        bool flag_ManualTraj = false;            // flag to control whether use the manual trajectory
        bool flag_MotorDynamics = false;         // flag to enable motor dynamics during simulation
        bool flag_ObsFilter = false;             // flag to control whether filter the observation value

        double ObsFilterFreq = 20;   // observation filter frequency
        double ObsFilterAlpha = 1.0; // observation filter parameter
        double stiffness = 40.0;     // PD control stiffness
        double stiffness_low = 5.0;  // PD control stiffness during swing phase
        double damping = 1.0;        // PD control damping
        double freq = 0;             // low pass through filter's frequency
        double contact_para = -0.5;  // contact shape function parameter, 1-exp(para*force)
        double EECoeff = 0.0;        // end effector reward coefficient
        double BodyPosCoeff = 0.0;   // keep body height reward coefficient
        double BodyAttiCoeff = 0.0;  // keep body attitude reward coefficient
        double JointMimicCoeff = 0.0;  // mimic the reference joint reward coefficient
        double VelKeepCoeff = 0.0;   // velocity follow coefficient
        double TorqueCoeff = 0.0;    // torque penalty coefficient
        double AttackProbability = 0.0;  // attack probability, =reward^3

        double cmd_update_param = 0.995;  // about 1s to reach the expected command
        // double cmd_update_param = 0.95;
        double command[3] = {0.0, 0.0, 0.0};  // update new command in reset()
        double command_filtered[3] = {0.0, 0.0, 0.0};  // store the filtered command
        double contact_[4] = {0.0, 0.0, 0.0, 0.0};     // store the contact state
        double contact_filtered[4] = {0.0, 0.0, 0.0, 0.0};  // filtered contacted information

        double Vx_max = 0.8;         // maximum body velocity along x direction (forward)
        double Vx_min = 0.0;         // minimum body velocity along x direction (forward)
        double Vy_max = 0.3;         // maximum body velocity along x direction (forward)
        double Vy_min = -0.3;         // minimum body velocity along x direction (forward)
        double omega_max = PI / 12.0;         // maximum body velocity along x direction (forward)
        double omega_min = -PI / 12.0;         // minimum body velocity along x direction (forward)

        unsigned long int itera = 0;
        int frame_idx = 0;    // frame of reference trajectory data
        int frame_max = 0;    // total frame number of reference trajectory
        int frame_len = 0;    // max frame of single explore
        double max_time = 0.0;
        double Lean_middle_front = 0.04;     // the distance of foot point moves symmetrically towards the center of front legs
        double Lean_middle_hind = 0.04;      // the distance of foot point moves symmetrically towards the center of hind legs

        // define disturbance of dynamics
        double mass_distrubance_ratio = 0.15;  // add 15% mass disturbance of robot
        double com_distrubance = 0.02;         // 0.02 m uncertainty of center of mass
        double calf_distrubance = 0.01;        // 0.01 m uncertainty of the calf length

        // Eigen::MatrixXf ref;
        Eigen::MatrixXf ref;
    };
}