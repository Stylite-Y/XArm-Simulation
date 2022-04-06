#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#if WIN32
#include <timeapi.h>
#endif

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  printf
  raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");
#if WIN32
    timeBeginPeriod(1); // for sleep_for function. windows default clock speed is 1/64 second. This sets it to 1ms.
#endif

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  world.addGround(0, "steel");
  auto sphere = world.addSphere(0.2, 1.0, "rubber");

  sphere->setPosition(0,0,5);

  world.setMaterialPairProp("steel", "rubber", 0.8, 0.15, 0.001);

  /// launch raisim server
  raisim::RaisimServer server(&world);
  server.launchServer();

  for (int i = 0; i < 500; i++) {
    raisim::MSLEEP(1);
    server.integrateWorldThreadSafe();
  }

  server.killServer();
}
