#pragma once
#include "states/agent.h"
#include <unordered_map>
#include <string>
#include <vector>
#include <random>

class QLearning {
    public:
        QLearning();
        void QLearningStep(Agent& env, std::unordered_map<std::string, std::vector<float>> &Q,
                           float gamma, float alpha, float epsilon,
                           int nActions);
    private:
        
        std::mt19937 rng;
};