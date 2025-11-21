#include "states/agent.h"
#include "ai/gradientdescent.h"

#include <SFML/System/Vector2.hpp>
#include <cmath>
#include <sstream>
#include <iostream>

Agent::Agent()
: positionX(0.0f),
positionY(0.0f),
speedForward(0.0f),
speedTurn(0.0f),
angle(0.0f),
prevPositionX(0.0f),
prevPositionY(0.0f),
prevSpeedForward(0.0f),
prevSpeedTurn(0.0f),
prevAngle(0.0f),
prevAction(0),
terminated(false),
truncated(false),
gradient(0),
prevGradient(0),
posIndex(0),
prevPosIndex(0) {}

void Agent::reset() {
    positionX = 0.0f;
    positionY = 0.0f;
    speedForward = 0.0f;
    speedTurn = 0.0f;
    angle = 0.0f;
    
    prevPositionX = 0.0f;
    prevPositionY = 0.0f;
    prevSpeedForward = 0.0f;
    prevSpeedTurn = 0.0f;
    prevAngle = 0.0f;
    prevAction = 0;
    
    gradient = 0;
    prevGradient = 0;
    
    posIndex = 0;
    prevPosIndex = 0;
    
    terminated = false;
    truncated = false;
}
void Agent::dumpQToFile(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing Q-table\n";
        return;
    }

    for (const auto& kv : Q) {
        const std::string& state = kv.first;
        const std::vector<float>& qvals = kv.second;

        out << state;
        for (float qv : qvals) {
            out << " " << std::setprecision(6) << qv;
        }
        out << "\n";
    }
}

void Agent::updatePosition(float x, float y) {
    positionX = x;
    positionY = y;
}

void Agent::updateSpeed(float forward, float turn) {
    speedForward = forward;
    speedTurn = turn;
}

void Agent::updateAngle(float a) {
    float ang = a;
    while (ang < 0.0f)   ang += 360.0f;
    while (ang >= 360.0f) ang -= 360.0f;
    angle = ang;
}

float Agent::getSpeedForward() {
    return speedForward;
}

float Agent::getSpeedTurn() {
    return speedTurn;
}

float Agent::getAngle() {
    return angle;
}

void Agent::updateGradient(int g) {
    gradient = g;
}

int Agent::getGradient() const {
    return gradient;
}

void Agent::render() {
    std::cout << getStateKey() << std::endl;
}
std::unordered_map<std::string, std::vector<float>>& Agent::getQ() {
    return Q;
}

void Agent::setQ(const std::unordered_map<std::string, std::vector<float>>& newQ) {
    Q = newQ;
}

std::string Agent::getStateKey() {
    std::ostringstream ss;
    ss << positionX << "|"
       << positionY << "|"
       << speedForward << "|"
       << speedTurn << "|"
       << angle;
    return ss.str();
}


std::string Agent::getPrevStateKey() {
    std::ostringstream ss;
    ss << prevPositionX << "|"
       << prevPositionY << "|"
       << prevSpeedForward << "|"
       << prevSpeedTurn << "|"
       << prevAngle;
    return ss.str();
}


int Agent::getPrevAction() {
    return prevAction;
}

void Agent::setPrevAction(int action) {
    prevAction = action;
}

int Agent::selectBestAction(int nActions) {
    std::string stateKey = getStateKey();

    auto it = Q.find(stateKey);
    if (it == Q.end()) {
        Q[stateKey] = std::vector<float>(nActions, 0.0f);
        it = Q.find(stateKey);
    } else if ((int)it->second.size() < nActions) {
        it->second.resize(nActions, 0.0f);
    }

    const std::vector<float>& qvals = it->second;

    int   bestAction = 0;
    float bestQ      = qvals.empty() ? 0.0f : qvals[0];

    for (int a = 1; a < nActions && a < (int)qvals.size(); ++a) {
        if (qvals[a] > bestQ) {
            bestQ = qvals[a];
            bestAction = a;
        }
    }

    return bestAction;
}

void Agent::setTerminated(bool t) {
    terminated = t;
}

void Agent::setTruncated(bool t) {
    truncated = t;
}
std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>>
Agent::step(int action) {
    std::vector<float> obs;
    obs.reserve(5);
    obs.push_back(positionX);
    obs.push_back(positionY);
    obs.push_back(speedForward);
    obs.push_back(speedTurn);
    obs.push_back(angle);

    std::map<std::string, float> info;
    info["position_x"]    = positionX;
    info["position_y"]    = positionY;
    info["speed_forward"] = speedForward;
    info["speed_turn"]    = speedTurn;
    info["angle_deg"]     = angle;

    float reward = 0.0f;
    sf::Vector2f pos(positionX, positionY);
    sf::Vector2f prevPos(prevPositionX, prevPositionY);
    sf::Vector2f disp = pos - prevPos;

    int curPosIndex = AIGradientDescent::getPositionValue(pos);
    float progressPos = static_cast<float>(curPosIndex - prevPosIndex);

    if (curPosIndex < 0) {
        reward -= 200.0f;
        reward -= std::fabs(speedForward) * 50.0f;

        prevPositionX    = positionX;
        prevPositionY    = positionY;
        prevSpeedForward = speedForward;
        prevSpeedTurn    = speedTurn;
        prevAngle        = angle;
        prevGradient     = gradient;
        prevPosIndex     = curPosIndex;
        posIndex         = curPosIndex;

        return std::make_tuple(obs, reward, terminated, truncated, info);
    }

    sf::Vector2f desiredDir = AIGradientDescent::getNextDirection(pos);
    float dmag = std::sqrt(desiredDir.x * desiredDir.x +
                           desiredDir.y * desiredDir.y);
    sf::Vector2f desiredUnit(0.0f, 0.0f);
    if (dmag > 1e-6f) {
        desiredUnit.x = desiredDir.x / dmag;
        desiredUnit.y = desiredDir.y / dmag;
    }

    float rad = angle * 3.1415926535f / 180.0f;
    sf::Vector2f heading(std::cos(rad), std::sin(rad));
    float hmag = std::sqrt(heading.x * heading.x +
                           heading.y * heading.y);
    if (hmag > 1e-6f) {
        heading.x /= hmag;
        heading.y /= hmag;
    }

    float cosAlign = heading.x * desiredUnit.x + heading.y * desiredUnit.y;
    if (cosAlign >  1.0f) cosAlign = 1.0f;
    if (cosAlign < -1.0f) cosAlign = -1.0f;

    float stepForward = disp.x * desiredUnit.x + disp.y * desiredUnit.y;

    reward += 200.0f * stepForward;

    if (stepForward < 0.0f) {
        reward += 100.0f * stepForward;
    }

    reward += 1.0f * cosAlign;

    reward += 0.5f * progressPos;

    reward -= 0.005f;

    prevPositionX    = positionX;
    prevPositionY    = positionY;
    prevSpeedForward = speedForward;
    prevSpeedTurn    = speedTurn;
    prevAngle        = angle;

    prevGradient     = gradient;
    prevPosIndex     = curPosIndex;
    posIndex         = curPosIndex;

    return std::make_tuple(obs, reward, terminated, truncated, info);
}
