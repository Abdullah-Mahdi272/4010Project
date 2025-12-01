#pragma once

#include <array>
#include <unordered_map>
#include <vector>
#include <random>
#include <string>

//step
struct MCAgentStep{
    int stateId;
    int action;
    double reward;
};

//actions for agent
struct MCActions{
    bool accel = false;
    bool left = false;
    bool right = false;
    
};

class MonteCarloAgent{
    public:
        static constexpr int NUM_ACTIONS = 3;

        MonteCarloAgent(double gamma = 0.99, double epsilon = 0.2);

        void startEpisode();

        void recordStep(int stateID, int action, double reward);

        void endEpisode(bool win);

        //epsilon greedy policy
        int selectAction(int stateId); 
        static MCActions actionFromIndex(int index);

        //ssaving and loaddding qtable
        bool save(const std::string& filename) const;
        bool load(const std::string& filename);
        int getEpisodeCount() const;

    private:
        double gamma;
        double epsilon;

        int episodeCount = 0;

        std::mt19937 rng;
        std::uniform_real_distribution<double> dist;

        //Q[s][a]
        using QRow = std::array<double, NUM_ACTIONS>;
        using RowCount = std::array<int, NUM_ACTIONS>;
        using VisitedRow = std::array<bool, NUM_ACTIONS>;

        std::unordered_map<int, QRow> Q;
        std::unordered_map<int, QRow> returnsSum;
        std::unordered_map<int, RowCount> returnsCount;

        std::vector<MCAgentStep> trajectory;

        QRow& getQRow(int stateId);
        QRow& getSumRow(int stateId);
        RowCount& getRowCount(int stateId);
        
};