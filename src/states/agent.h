#pragma once
#include <iostream>
#include <cmath>
#include <tuple>
#include <vector>
#include <map>
#include <string>

class Agent {
public:
    Agent();
    
    // Update agent state from game
    void updatePosition(float x, float y);
    void updateSpeed(float forward, float turn);
    void updateAngle(float angle);
    void updateGameState(int gradient, int lap, int rank, int coins);
    
    // Get current state
    float getSpeedForward();
    float getSpeedTurn();
    float getAngle();
    
    // Output state to Python environment
    void outputState(float raceTime, int nextSplitIndex);
    void outputTrackInfo(int maxGradient, float trackWidth, float trackHeight);
    void outputEpisodeStart();
    void outputRaceEnd(int finalRank, float finalTime);
    
    // Read action from Python environment
    bool readAction(int& accelerate, int& brake, int& left, int& right);
    
    // Gym-style interface
    void reset();
    std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>> step(int);
    
    void setTerminated(bool t);
    void setTruncated(bool t);
    
    // Check episode state
    inline bool isTerminated() const { return terminated; }
    inline bool isTruncated() const { return truncated; }

private:
    // Position and movement
    float positionX;
    float positionY;
    float speedForward;
    float speedTurn;
    float angle;
    
    // Previous state for reward calculation
    float prevPositionX;
    float prevPositionY;
    float prevSpeedForward;
    float prevSpeedTurn;
    float prevAngle;
    static int prevGradient;
    static int prevRank;
    
    // Game state
    int currentGradient;
    int currentLap;
    int currentRank;
    int currentCoins;
    
    // Episode state
    bool terminated;
    bool truncated;
};
