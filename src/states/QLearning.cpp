#include "QLearning.h"
#include <random>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <iostream>

QLearning::QLearning() : rng(std::random_device{}()) {
    std::cout << "QLearning constructed at " << this << std::endl;
}


void QLearning::QLearningStep(Agent& env, std::unordered_map<std::string, std::vector<float>> &Q,
                   float gamma, float alpha, float epsilon,
                   int nActions) 
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> actionDist(0, nActions - 1);

    std::string stateKey = env.getPrevStateKey();
    if (Q.find(stateKey) == Q.end()) {
        Q[stateKey] = std::vector<float>(nActions);
        for (float& qv : Q[stateKey]) qv = dist(rng);
    }
    
    int action = env.getPrevAction();

    auto [obs, reward, terminated, truncated, info] = env.step(action);

    std::string nextStateKey = env.getStateKey();
    if (Q.find(nextStateKey) == Q.end()) {
        Q[nextStateKey] = std::vector<float>(nActions);
        for (float& qv : Q[nextStateKey]) qv = dist(rng);
    }

    auto& q_curr = Q[stateKey];
    auto& q_next = Q[nextStateKey];
    int bestNext = std::distance(q_next.begin(), std::max_element(q_next.begin(), q_next.end()));
    q_curr[action] += alpha * (reward + gamma * q_next[bestNext] - q_curr[action]);
}
