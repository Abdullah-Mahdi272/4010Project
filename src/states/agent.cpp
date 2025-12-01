#include "states/agent.h"
#include <iostream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

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
      prevSpeedTurn(0.0f),
      prevAngle(0.0f),
      terminated(false),
      truncated(false),
      currentGradient(0),
      currentLap(1),
      currentRank(8),
      currentCoins(0) {
    
    // Force unbuffered stdout/stderr for immediate communication
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);
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
    // normalize to [0, 360)
    float ang = a;
    while (ang < 0.0f) ang += 360.0f;
    while (ang >= 360.0f) ang -= 360.0f;
    angle = ang;
}

void Agent::updateGameState(int gradient, int lap, int rank, int coins) {
    currentGradient = gradient;
    currentLap = lap;
    currentRank = rank;
    currentCoins = coins;
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

void Agent::outputState(float raceTime, int nextSplitIndex) {
    // Output in format for Python environment to parse
    // Format: RL_STATE|time|gradient|lap|split|posX|posY|speed|turnSpeed|angle|rank|coins
    std::cout << "RL_STATE|"
              << raceTime << "|"
              << currentGradient << "|"
              << currentLap << "|"
              << nextSplitIndex << "|"
              << positionX << "|"
              << positionY << "|"
              << speedForward << "|"
              << speedTurn << "|"
              << angle << "|"
              << currentRank << "|"
              << currentCoins << std::endl;
    std::cout.flush();  // Ensure immediate output
}

void Agent::outputTrackInfo(int maxGradient, float trackWidth, float trackHeight) {
    // Output track information for environment initialization
    std::cout << "TRACK_INFO|"
              << maxGradient << "|"
              << trackWidth << "|"
              << trackHeight << std::endl;
    std::cout.flush();
}

void Agent::outputEpisodeStart() {
    // Signal that the episode is starting and Python can begin
    std::cout << "EPISODE_START" << std::endl;
    std::cout.flush();
}

void Agent::outputRaceEnd(int finalRank, float finalTime) {
    // Output race completion
    std::cout << "RACE_END|"
              << finalRank << "|"
              << finalTime << std::endl;
    std::cout.flush();
}

bool Agent::readAction(int& accelerate, int& brake, int& left, int& right) {
    // Non-blocking read from stdin
    // This allows the game to continue even if Python hasn't sent an action yet
    
    // Save current stdin flags
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    if (flags == -1) {
        return false;  // Error getting flags
    }
    
    // Set stdin to non-blocking mode
    if (fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK) == -1) {
        return false;  // Error setting non-blocking
    }
    
    // Try to read a line
    std::string line;
    bool success = false;
    
    if (std::getline(std::cin, line)) {
        if (line.find("ACTION|") == 0) {
            // Parse action: ACTION|accel|brake|left|right
            std::stringstream ss(line.substr(7)); // Skip "ACTION|"
            char delim;
            
            if (ss >> accelerate >> delim >> brake >> delim >> left >> delim >> right) {
                success = true;
            }
        } else if (line == "RESET") {
            // Reset command - reset agent state
            reset();
        } else if (line == "GET_TRACK_INFO") {
            // Track info request - will be handled elsewhere
        }
    }
    
    // Restore original stdin flags (blocking mode)
    fcntl(STDIN_FILENO, F_SETFL, flags);
    
    // Clear any error flags on cin
    if (std::cin.eof()) {
        std::cin.clear();
    }
    
    return success;
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
    prevSpeedTurn = 0.0f;
    prevAngle = 0.0f;
    terminated = false;
    truncated = false;
    currentGradient = 0;
    currentLap = 1;
    currentRank = 8;
    currentCoins = 0;
}

void Agent::setTerminated(bool t) {
    terminated = t;
}

void Agent::setTruncated(bool t) {
    truncated = t;
}

std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> Agent::step(int action) {
    // Create observation vector
    std::vector<float> obs;
    obs.reserve(11);
    obs.push_back(positionX);
    obs.push_back(positionY);
    obs.push_back(speedForward);
    obs.push_back(speedTurn);
    obs.push_back(angle);
    obs.push_back(static_cast<float>(currentGradient));
    obs.push_back(static_cast<float>(currentLap));
    obs.push_back(static_cast<float>(currentRank));
    obs.push_back(static_cast<float>(currentCoins));
    
    // Calculate reward based on progress
    float reward = 0.0f;
    
    // Speed reward
    reward += 0.1f * speedForward;
    
    // Progress reward (gradient decreases as you progress)
    float gradientProgress = prevGradient - currentGradient;
    if (gradientProgress > 0) {
        reward += gradientProgress * 0.01f;
    } else if (gradientProgress < -10) {
        reward -= 10.0f; // Penalty for going backwards
    }
    
    // Rank improvement reward
    if (currentRank < prevRank) {
        reward += 5.0f * (prevRank - currentRank);
    } else if (currentRank > prevRank) {
        reward -= 2.0f * (currentRank - prevRank);
    }
    
    // Update previous values
    prevPositionX = positionX;
    prevPositionY = positionY;
    prevSpeedForward = speedForward;
    prevSpeedTurn = speedTurn;
    prevAngle = angle;
    prevGradient = currentGradient;
    prevRank = currentRank;
    
    // Create info dictionary
    std::map<std::string, float> info;
    info["position_x"] = positionX;
    info["position_y"] = positionY;
    info["speed_forward"] = speedForward;
    info["speed_turn"] = speedTurn;
    info["angle_deg"] = angle;
    info["gradient"] = static_cast<float>(currentGradient);
    info["lap"] = static_cast<float>(currentLap);
    info["rank"] = static_cast<float>(currentRank);
    info["coins"] = static_cast<float>(currentCoins);
    
    return std::make_tuple(obs, reward, terminated, truncated, info);
}

int Agent::prevGradient = 0;
int Agent::prevRank = 8;
