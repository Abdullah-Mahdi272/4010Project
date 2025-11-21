#pragma once

#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <fstream>
#include <iomanip>
class Agent {
public:
    Agent();
    void reset();
    void dumpQToFile(const std::string& filename) const;
    void updatePosition(float x, float y);
    void updateSpeed(float forward, float turn);
    void updateAngle(float a);
    void updateGradient(int g);
    int  getGradient() const;

    float getSpeedForward();
    float getSpeedTurn();
    float getAngle();

    void render();
    std::unordered_map<std::string, std::vector<float>>& getQ();
    void setQ(const std::unordered_map<std::string, std::vector<float>>& newQ);

    std::string getStateKey();
    std::string getPrevStateKey();

    int  getPrevAction();
    void setPrevAction(int action);

    int selectBestAction(int nActions);

    void setTerminated(bool t);
    void setTruncated(bool t);
    float getPositionX() const { return positionX; }
    float getPositionY() const { return positionY; }

    std::tuple<std::vector<float>, float, bool, bool, std::map<std::string, float>>
    step(int action);

private:
    
    float positionX;
    float positionY;
    float speedForward;
    float speedTurn;
    float angle;

    float prevPositionX;
    float prevPositionY;
    float prevSpeedForward;
    float prevSpeedTurn;
    float prevAngle;

    int   prevAction;

    bool  terminated;
    bool  truncated;

    int gradient;
    int prevGradient;

    int posIndex;
    int prevPosIndex;

    std::unordered_map<std::string, std::vector<float>> Q;
};
