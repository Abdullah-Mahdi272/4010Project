#include "MonteCarloAgent.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

//constructor
MonteCarloAgent::MonteCarloAgent(double gamma, double epsilon) : gamma(gamma), epsilon(epsilon), rng(std::random_device{}()), dist(0.0, 1.0){}

//return q values array for state
MonteCarloAgent::QRow& MonteCarloAgent::getQRow(int stateId) {
    auto it = Q.find(stateId);
    //if state doesnt exist
    if (it == Q.end()) {
        //make q values
        QRow q{};
        q.fill(0.0);
        Q[stateId] = q;

        //sums
        QRow s{};
        s.fill(0.0);
        returnsSum[stateId] = s;

        //visited rows
        RowCount c{};
        c.fill(0);
        returnsCount[stateId] = c;

        return Q[stateId];
    }
    //return qvalues for sstate
    return it->second;
}

//return cumulative returns row 
MonteCarloAgent::QRow& MonteCarloAgent::getSumRow(int stateId){
    auto it = returnsSum.find(stateId);
    //if it doesnt exist
    if(it == returnsSum.end()){
        QRow s{};
        s.fill(0.0);
        returnsSum[stateId] = s;
        return returnsSum[stateId];

    }
    //return sum for state
    return it->second;

}

//returns number of times each SA pair has been upddated
MonteCarloAgent::RowCount& MonteCarloAgent::getRowCount(int stateId) {
    auto it = returnsCount.find(stateId);
    if (it == returnsCount.end()) {
        RowCount c{};
        c.fill(0);
        returnsCount[stateId] = c;
        return returnsCount[stateId];
    }
    return it->second;
}

//start episode, clear trajectory
void MonteCarloAgent::startEpisode() {
    trajectory.clear();
}

//choose an action with epsilon greedy
int MonteCarloAgent::selectAction(int stateId) {
    QRow &q = getQRow(stateId);

    //explore
    if (dist(rng) < epsilon) {
        std::uniform_int_distribution<int> adist(0, NUM_ACTIONS - 1);
        return adist(rng);
    } 
    //exploit
    else {
        int bestIdx = 0;
        double bestVal = q[0];
        for (int i = 1; i < NUM_ACTIONS; ++i) {
            if (q[i] > bestVal) {
                bestVal = q[i];
                bestIdx = i;
            }
    }
        //return action index with highest q value
        return bestIdx;
    }
}

//maps action inddex to action
MCActions MonteCarloAgent::actionFromIndex(int idx) {
    MCActions d;
    switch (idx) {
        case 0:
            //drive straight
            d.accel = true;
            break;

        case 1:
            //turn left
            d.accel = true;
            d.left  = true;
            break;

        case 2:
            //turn right
            d.accel = true;
            d.right = true;
            break;

        default:
            //drive straight
            d.accel = true;
            break;
    }

    return d;
}

//stores step in episodde for learning
void MonteCarloAgent::recordStep(int stateId, int action, double reward) {
    MCAgentStep s;
    s.stateId = stateId;
    s.action  = action;
    s.reward  = reward;
    trajectory.push_back(s);
}

//ends current race (episode)
void MonteCarloAgent::endEpisode(bool win) {
    if (trajectory.empty()) return;

    //testing
    //compute total returns
    double totalReturn = 0.0;
    {
        double tempG = win ? 50.0 : 0.0;
        for (int t = (int)trajectory.size() - 1; t >= 0; --t) {
            tempG = gamma * tempG + trajectory[t].reward;
        }
        totalReturn = tempG;
    }

    //increment episode
    episodeCount++;
    //print info
    std::cout << "[MC] Episode " << episodeCount<< " finished. epsilon = " << epsilon << "\n";
    std::cout << "[MC] Episode finished. steps=" << trajectory.size()<< " return≈" << totalReturn<< " epsilon=" << epsilon << "\n";

    //bonus if in top 3
    double G = 0.0;
    if (win) {
        G += 50.0;
    }

    //track actions updated for each state
    std::unordered_map<int, VisitedRow> visited;

    //work backwords thru episodde
    for (int t = static_cast<int>(trajectory.size()) - 1; t >= 0; --t) {
        auto &step = trajectory[t];
        //monte carlo return update
        G = gamma * G + step.reward;

        //make sure rows exxists
        auto &visRow = visited[step.stateId];
        if (visRow[0] == false && visRow[1] == false && visRow[2] == false &&
            visRow[3] == false && visRow[4] == false && visRow[5] == false &&
            visRow[6] == false) {
            visRow.fill(false);
        }

        //update for first time SA pairs
        if (!visRow[step.action]) {
            visRow[step.action] = true;

            QRow     &sumRow   = getSumRow(step.stateId);
            RowCount &countRow = getRowCount(step.stateId);
            QRow     &qRow     = getQRow(step.stateId);

            sumRow[step.action]   += G;
            countRow[step.action] += 1;
            qRow[step.action]      = sumRow[step.action] / countRow[step.action];
        }
    }

    

    trajectory.clear();

    //epsilon decay
    epsilon = std::max(0.01, epsilon * 0.999);
    
}

//save qtable and update epsilon to txt file
bool MonteCarloAgent::save(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) return false;


    //stateId, epsilon, q0 - q6
    out << "# epsilon " << epsilon << "\n";
    for (const auto &entry : Q) {
        int stateId = entry.first;
        const QRow &qRow = entry.second;
        out << stateId;
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            out << " " << qRow[a];
        }
        out << "\n";
    }

    return true;
}

//load file
bool MonteCarloAgent::load(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) return false;

    Q.clear();
    returnsSum.clear();
    returnsCount.clear();

    std::string token;
    in >> token;
    if (token == "#") {
        std::string epsWord;
        in >> epsWord >> epsilon; 
    } else {
        in.clear();
        in.seekg(0);
    }

    int stateId;
    while (in >> stateId) {
        QRow qRow{};
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            in >> qRow[a];
        }
        Q[stateId] = qRow;

        //init return sum count
        QRow s{};
        s.fill(0.0);
        returnsSum[stateId] = s;

        RowCount c{};
        //pretend to see action once to avoid /0
        c.fill(1);
        returnsCount[stateId] = c;
    }

    std::cout << "[MC] Loaded policy from " << filename
              << " with " << Q.size() << " states.\n";
    return true;
}

//return current episode
int MonteCarloAgent::getEpisodeCount() const { return episodeCount; }