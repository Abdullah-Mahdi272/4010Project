#include "states/agent.h"

static constexpr float PI_F = 3.14159265358979323846f;

Agent::Agent()
    : positionX(0.0f),
      positionY(0.0f),
      speedForward(0.0f),
      speedTurn(0.0f),
      angle(0.0f),
      prevPositionX(0.0f),
      prevPositionY(0.0f),
      prevSpeedForward(0.0f),
      terminated(false),
      truncated(false) {}

void Agent::updatePosition(float x, float y) {
    positionX = x;
    positionY = y;
}

void Agent::updateSpeed(float forward, float turn) {
    speedForward = forward;
    speedTurn = turn;
}

void Agent::updateAngle(float a) {
    // normalize to [0, 360)
    float ang = a;
    while (ang < 0.0f) ang += 360.0f;
    while (ang >= 360.0f) ang -= 360.0f;
    angle = ang;
}

float Agent::getSpeedForward() { 
    return speedForward;
}

float Agent::getSpeedTurn() {
    return speedTurn; 
}

float Agent::getAngle(){ 
    return angle; 
}

void Agent::render(){
    std::cout<< "Agent angle updated to: " << angle << std::endl;
    std::cout<< "Speed forward updated to: " << speedForward << std::endl;
    std::cout<< "Speed turn updated to: " << speedTurn << std::endl;
    std::cout<< "Position updated to: " << positionX << "," << positionY << std::endl;
}

void Agent::reset() {
    positionX = 0.0f;
    positionY = 0.0f;
    speedForward = 0.0f;
    speedTurn = 0.0f;
    angle = 0.0f;
    prevPositionX = 0.0f;
    prevPositionY = 0.0f;
    prevSpeedForward = 0.0f;
    terminated = false;
    truncated = false;
}

void Agent::setTerminated(bool t) {
    terminated = t;
}

void Agent::setTruncated(bool t) {
    truncated = t;
}

std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> Agent::step(int action) {
    std::vector<float> obs;
    obs.reserve(5);
    obs.push_back(positionX);
    obs.push_back(positionY);
    obs.push_back(speedForward);
    obs.push_back(speedTurn);
    obs.push_back(angle);
    float reward = 0.0f;

    std::map<std::string, float> info;
    info["position_x"] = positionX;
    info["position_y"] = positionY;
    info["speed_forward"] = speedForward;
    info["speed_turn"] = speedTurn;
    info["angle_deg"] = angle;
    float prevSF = prevSpeedForward;

    const float tol = 1e-3f;

    if (speedForward + tol >= prevSF) {
        reward = 1.0f;
    } else {
        reward = -1.0f;
    }

    prevPositionX = positionX;
    prevPositionY = positionY;
    prevSpeedForward = speedForward;
    prevAngle = angle;
    prevSpeedTurn = speedTurn;


    return std::make_tuple(obs, reward, terminated, truncated, info);
}