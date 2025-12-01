#include "race.h"
std::unique_ptr<MonteCarloAgent> StateRace::mcAgent = nullptr;
// #define DEBUG_POSITION_RANKING  // uncomment to show positions in ranking

//monte carlo
//calculate progress
static float getProgress(const Driver& d){
    //get maximum gradient index (how many parts of track, high early, low later)
    int maxG = AIGradientDescent::MAX_POSITION_MATRIX;
    //num laps
    int lap = d.getLaps();
    //current pqrt of track
    int gradient = d.getLastGradient();

    //if gradient less than 0
    if (gradient < 0){
        gradient = 0;
    }
    //if gradient indexx out of bounds
    if (gradient > maxG) {
        gradient = maxG;
    }
    //return progresss
    return static_cast<float>(lap * (maxG + 1) + (maxG - gradient));

}

//monte carlo
//makes current state into stateId (ddiscretize state space)
static int makeStateId(const DriverPtr& driver){
    //get max gradient index
    int maxG = AIGradientDescent::MAX_POSITION_MATRIX;

    //get lap that agent is on
    int lap = driver->getLaps();
    //if lap less than 0, set to 0
    if(lap < 0){
        lap = 0;
    }

    //if lap is > 3, set to 3 max
    if(lap > 3){
        lap = 3;
    }

    //get graddient of agent
    int grad = driver->getLastGradient();
    //if gradient out of bounds, set in bounds
    if (grad < 0) {
        grad = 0;
    }
    if (grad > maxG) {
        grad = maxG;
    }

    //gradient bins to split up track (reasonable number of sections)
    const int GRAD_BINS = 20;

    
    int gradBin = 0;
    if(maxG > 0){
        //convert actual gradient to reasonable number
        gradBin = grad * GRAD_BINS / (maxG + 1);
        //if it goes out of bounds
        if(gradBin >= GRAD_BINS){
            gradBin = GRAD_BINS - 1;
        }
    }

    //speed bins to split different speed values into reasonable number
    float speed = driver->speedForward;
    float maxSpeed = driver->vehicle->maxNormalLinearSpeed;

    //5 speeds
    const int SPEED_BINS = 5;
    int speedBin = 0;


    if(maxSpeed > 0.0f){
        //if speed out of bounds
        if(speed < 0.0f){
            speed = 0.0f;
        }

        if(speed > maxSpeed){
            speed = maxSpeed;
        }
        //map speed to speed bin
        speedBin = static_cast<int>(speed * SPEED_BINS / maxSpeed);

        //if speed bin out of bounds
        if(speedBin >= SPEED_BINS){
            speedBin = SPEED_BINS - 1;
        }
    }

    //combine into one state ID [lap][gradient][speed]
    int stateId = lap * (GRAD_BINS * SPEED_BINS) + gradBin * SPEED_BINS + speedBin;

    return stateId;
}

const sf::Time StateRace::TIME_BETWEEN_ITEM_CHECKS =
    sf::seconds(1.0f) / (float)Item::UPDATES_PER_SECOND;

const sf::Time StateRace::WAIT_FOR_PC_LAST_PLACE = sf::seconds(5.0f);

StateRace::~StateRace() {
    if (agent != nullptr) {
        delete agent;
        agent = nullptr;
    }
}

void StateRace::init() {
    pushedPauseThisFrame = false;
    StateRace::currentTime = sf::Time::Zero;
    nextItemCheck = sf::Time::Zero;
    waitForPCTime = sf::Time::Zero;
    splitsInitialized = false;
    collidedThisFrame = false;

    //random
    //false = player control
    random_enabled = true;
    randomAI = std::make_unique<RandomAgent>();

    //monte carlo
    mc_enabled = true;
    if(mc_enabled){
        if(!mcAgent){
            mcAgent = std::make_unique<MonteCarloAgent>(0.99, 0.2);
            mcAgent->load("mc_policy.txt");
            
        }
        lastProgress = 0.0;
        noProgressFrames = 0;
    }
    
    // ensure we have an Agent instance for this race
    if (agent == nullptr) {
        agent = new Agent();
    }

    // reset agent stored state at race start
    if (agent != nullptr) {
        agent->reset();
    }
}

//get player info
PlayerInfo StateRace::getPlayerInfo(const DriverPtr& d){
    PlayerInfo info;

    info.progress = getProgress(*d);
    info.speed = d->speedForward;
    info.angle = d->posAngle;
    info.lap = d->getLaps();
    info.rank = d->rank;
    info.onGround = (d->height == 0.0f);
    info.canDrive = d->canDrive();

    return info;

}

void StateRace::handleEvent(const sf::Event& event) {
    // items
    if (Input::pressed(Key::ITEM_FRONT, event) && player->canDrive()) {
        Item::useItem(player, positions, true);
    }
    if (Input::pressed(Key::ITEM_BACK, event) && player->canDrive()) {
        Item::useItem(player, positions, false);
    }

    // drifting
    if (Input::pressed(Key::DRIFT, event) && player->canDrive() &&
        !driftPressed) {
        driftPressed = true;
        player->shortJump();
    }
    if (Input::released(Key::DRIFT, event)) {
        driftPressed = false;
    }

    // pause menu
    if (Input::pressed(Key::PAUSE, event) && !pushedPauseThisFrame) {
        pushedPauseThisFrame = true;
        // call draw and store so we can draw it over the screen
        sf::RenderTexture render;
        sf::Vector2u windowSize = game.getWindow().getSize();
        render.create(windowSize.x, windowSize.y);
        fixedUpdate(sf::Time::Zero);
        draw(render);
        game.pushState(StatePtr(new StateRacePause(game, render)));
    }
}

bool StateRace::fixedUpdate(const sf::Time& deltaTime) {
    // don't update if we already popped
    if (raceFinished) {
        return true;
    }

    //increase episode time as number of episodes increase
    if (mc_enabled && mcAgent) {
    int ep = mcAgent->getEpisodeCount();
        if (ep < 200)
            maxEpisodeTimeSecs = 15.0;
        else if (ep < 500)
            maxEpisodeTimeSecs = 30.0;
        else if (ep < 1000)
            maxEpisodeTimeSecs = 60.0;
        else
            maxEpisodeTimeSecs = 120.0;
    }

    // update global time
    currentTime += deltaTime;
    pushedPauseThisFrame = false;

    // Initialize splits after first update when gradient data is available
    if (!splitsInitialized && player->getLastGradient() != -1) {
        // Get max gradient from AI system
        int maxGradient = AIGradientDescent::MAX_POSITION_MATRIX;
        Gui::initializeSplits(maxGradient);
        splitsInitialized = true;
    }

    // Map object updates
    for (unsigned int i = 0; i < drivers.size(); i++) {
        DriverPtr& driver = drivers[i];
        // Player position updates
        driver->update(deltaTime);
        Audio::updateEngine(i, driver->position, driver->height,
                            driver->speedForward, driver->speedTurn);
        
        //get player info every frame for monte carlo                 
        if(driver.get() == player.get()){
            PlayerInfo info = getPlayerInfo(driver);

            //test print
            /*std::cout << "=== Player Info ===\n";
            std::cout << "Position: (" << driver->position.x 
                    << ", " << driver->position.y << ")\n";
            std::cout << "Speed Forward: " << driver->speedForward << "\n";
            std::cout << "Speed Turn: "    << driver->speedTurn << "\n";
            std::cout << "Angle: "         << driver->posAngle << "\n";
            std::cout << "Lap: "           << driver->getLaps() << "\n";
            std::cout << "Gradient: "      << driver->getLastGradient() << "\n";
            std::cout << "Rank: "          << driver->rank << "\n";
            std::cout << "Height: "        << driver->height << "\n";
            std::cout << "Can Drive: "     << driver->canDrive() << "\n";*/
        }
                            

        //random agent
        //if random is enabled and if this driver is the player
        if (mc_enabled && mcAgent && driver == player) {
            //make state ID
            int stateId = makeStateId(driver);

            //get action from monte carlo
            int actionIdx = mcAgent->selectAction(stateId);
            MCActions d = MonteCarloAgent::actionFromIndex(actionIdx);

    
        //if plauer os on the ground
        if (driver->height == 0.0f) {
            //if accelerate
            if (d.accel) {
                //accelerate smoothly until max speed
                float f = driver->vehicle->motorAcceleration * 0.5f;
                driver->speedForward = std::min(driver->speedForward + f * deltaTime.asSeconds(), driver->vehicle->maxNormalLinearSpeed);
            }

            //no braking for now
            //braking
            /*if (d.brake) {
                //accelerate backwards 
                float b = driver->vehicle->motorAcceleration * 0.6f;
                driver->speedForward = std::max(driver->speedForward - b * deltaTime.asSeconds(), 0.0f);
            }*/
        }

        //turning
        //if turning left
        if (d.left && !d.right) {
            driver->speedTurn = -driver->vehicle->maxTurningAngularSpeed * 0.5f;
        } 
        //if turning right
        else if (d.right && !d.left) {
            driver->speedTurn =  driver->vehicle->maxTurningAngularSpeed * 0.5f;
        } 
        //if not turning (going back to moving straight)
        else {
            //reduce turning
            driver->speedTurn /= 1.5f;
        }

        //no jump or item for now
        //drift (jump)
        //if driver is able to drive and is on the ground
        /*if (d.drift && driver->canDrive() && driver->height == 0.0f) {
            driver->shortJump();
        }

        //if random agent decides to use item AND if player is able to drive AND if player has item
        if (d.item && driver->canDrive() && driver->getPowerUp() != PowerUps::NONE) {
            //uses item to front (false would be backwards)
            Item::useItem(driver, positions, true);
        }*/


        //rewards
        if (!rewardInit) {
            // wait until gradient is valid before starting reward
            if (driver->getLastGradient() >= 0) {
                //initalize rewards
                lastRewardProgress = getProgress(*driver);
                rewardInit = true;
            }
        } 
        //if rewards are initalized
        else {
            //init zero reward 
            double reward = 0.0;

            //get progression in track
            double newProg   = getProgress(*driver);
            //progress made since last frame
            double progMade = newProg - lastRewardProgress;
            //used for next step
            lastRewardProgress = newProg;

            //forward progress reward
            reward += 5.0 * progMade;  

            //if setback (out of bounds)
            if (progMade < -15.0) {
                //punish and endd episode
                reward -= 200.0;
                std::cout << "OUT OF BOUNDS!\n";
                
                //recordd last step
                mcAgent->recordStep(stateId, actionIdx, reward);
                mcAgent->endEpisode(false);
                mcAgent->save("mc_policy.txt");

                raceFinished = true;
                game.popState();
                return true;
            }

            //going backwards punishment
            if (driver->speedForward < -0.5f) {
                reward -= 3.0;
            }

            

            //if agent is accelerating but making no progress (stuck)
            if (d.accel && progMade <= 0.0) {
                //increment frames with no progress by 1 (consecutive no progress)
                noProgressFrames++;
                
                //if agent gets stuck for more than 6 secondds
                if (noProgressFrames >= 360) {
                    //punish and endd episodde
                    reward -= 100.0;  
                    std::cout<<"STUCK! PUNISHMENT!";  
                    //reset no progress frames       
                    noProgressFrames = 0; 

                    //record last srep
                    mcAgent->recordStep(stateId, actionIdx, reward);
                    mcAgent->endEpisode(false);
                    mcAgent->save("mc_policy.txt");

                    raceFinished = true;
                    game.popState();
                    return true;
                    }
            }

            //if no longer stuck
            else{
                //reset frames with no progress
                noProgressFrames = 0;
            }

            //record values for state/action
            mcAgent->recordStep(stateId, actionIdx, reward);
        }

        //end of rewards
        //end of randdom agent
}
        
        if (driver == player) {
            // Update split timer for player WITH REAL DATA
            if (splitsInitialized) {
                Gui::updateSplits(
                    currentTime,
                    driver->getLastGradient(),
                    driver->getLaps(),
                    deltaTime,
                    driver->position,
                    driver->speedForward,
                    driver->speedTurn,
                    driver->posAngle
                );
            }

            // Agent code (if you still need it)
            /*if (agent != nullptr) {
                std::cout << "=== AGENT STEP ===" << std::endl;
                agent->updatePosition(driver->position.x, driver->position.y);
                agent->updateSpeed(driver->speedForward, driver->speedTurn);
                agent->updateAngle(driver->posAngle);
                agent->render();
                if (raceFinished) {
                    agent->setTerminated(true);
                }
                
                auto [obs, reward, terminated, truncated, info] = agent->step(0);

            }*/
        }
    }
    
    // check if AI should use its items
    if (currentTime > nextItemCheck) {
        nextItemCheck = currentTime + TIME_BETWEEN_ITEM_CHECKS;
        for (const DriverPtr& driver : drivers) {
            if (driver != player && driver->getPowerUp() != PowerUps::NONE) {
                float r = rand() / (float)RAND_MAX;
                AIItemProb prob = Item::getUseProbability(driver, positions);
                if (r < std::get<0>(prob) / driver->itemProbModifier) {
                    Item::useItem(driver, positions, std::get<1>(prob));
                }
            }
        }
    }
    Map::updateObjects(deltaTime);
    Audio::updateListener(player->position, player->posAngle, player->height);

    // Collision updates
    // Register all objects for fast detection
    CollisionHashMap::resetDynamic();
    Map::registerItemObjects();
    for (const DriverPtr& driver : drivers) {
        CollisionHashMap::registerDynamic(driver);
    }

    // Detect collisions with players
    CollisionData data;
    for (const DriverPtr& driver : drivers) {
        if (CollisionHashMap::collide(driver, data)) {
            collidedThisFrame = true;
            driver->collisionMomentum = data.momentum;
            driver->speedForward *= data.speedFactor;
            driver->speedTurn *= data.speedFactor;
            switch (data.type) {
                case CollisionType::HIT:
                    driver->applyHit();
                    break;
                case CollisionType::SMASH:
                    driver->applySmash();
                    break;
                default:
                    if (!driver->isImmune()) {
                        driver->addCoin(-1);
                    }
                    if (driver == player && driver->canDrive() &&
                        driver->speedForward != 0.0f) {
                        Audio::play(SFX::CIRCUIT_COLLISION_PIPE);
                    }
                    Map::addEffectSparkles(driver->position);
                    break;
            }
        }
    }

    // Ranking updates - last gradient contains
    auto hasntFinishedBegin = positions.begin();
    // don't sort drivers that have already finished the circuit
    while ((*hasntFinishedBegin)->getLaps() > NUM_LAPS_IN_CIRCUIT &&
           hasntFinishedBegin < positions.end()) {
        ++hasntFinishedBegin;
    }
    std::sort(hasntFinishedBegin, positions.end(),
              [](const Driver* lhs, const Driver* rhs) {
                  // returns true if player A is ahead of B
                  if (lhs->getLaps() == rhs->getLaps()) {
                      return lhs->getLastGradient() < rhs->getLastGradient();
                  } else {
                      return lhs->getLaps() > rhs->getLaps();
                  }
              });
    // find current player and update GUI
    for (unsigned int i = 0; i < positions.size(); i++) {
        positions[i]->rank = i;
        if (positions[i] == player.get()) {
            Gui::setRanking(i + 1);
        }
    }

    // UI updates
    Lakitu::update(deltaTime);
    bool hasChanged = FloorObject::applyAllChanges();
    if (hasChanged) {
        Map::updateMinimap();
    }

    EndRanks::update(deltaTime);
    Gui::update(deltaTime);

    // start time counter if all 7 AI finished before the player
    if (waitForPCTime == sf::Time::Zero &&
        positions[positions.size() - 2]->getLaps() > NUM_LAPS_IN_CIRCUIT) {
        waitForPCTime = currentTime + WAIT_FOR_PC_LAST_PLACE;
    }

    //monte carlo
    //if episode time limit is reached
    if (mc_enabled && mcAgent && currentTime.asSeconds() > maxEpisodeTimeSecs) {
        std::cout << "END OF EPISODE!\n";
        mcAgent->endEpisode(false);
        mcAgent->save("mc_policy.txt");

        raceFinished = true;
        game.popState();
        return true;
    }   
    // end the race if player has finished or all other AI have finished and the
    // grace time has ended
    if ((player->getLaps() > NUM_LAPS_IN_CIRCUIT ||
         (waitForPCTime != sf::Time::Zero && currentTime > waitForPCTime &&
          player->canDrive() && player->height == 0.0f)) &&
        !raceFinished) {
        raceFinished = true;

        CollisionHashMap::resetStatic();
        CollisionHashMap::resetDynamic();
        Audio::stopSFX();
        Audio::play(SFX::CIRCUIT_GOAL_END);
        Audio::setEnginesVolume(75.0f);
        Gui::stopEffects();

        if (player->getRank() <= 3) {
            Audio::play(SFX::CIRCUIT_END_VICTORY);
            Audio::play(Music::CIRCUIT_PLAYER_WIN);
        } else {
            Audio::play(SFX::CIRCUIT_END_DEFEAT);
            Audio::play(Music::CIRCUIT_PLAYER_LOSE);
        }

        for (const DriverPtr& driver : drivers) {
            driver->endRaceAndReset();
        }

        Lakitu::showFinish();
        Gui::endRace();
        player->controlType = DriverControlType::AI_GRADIENT;


        //monte carlo
        if(mc_enabled && mcAgent){
            bool win = (player->getRank() <= 3);
            mcAgent->endEpisode(win);
            mcAgent->save("mc_policy.txt");
        }

        game.popState();
    }

    return true;
}


void StateRace::draw(sf::RenderTarget& window) {
    // scale
    static constexpr const float NORMAL_WIDTH = 512.0f;
    const float scale = window.getSize().x / NORMAL_WIDTH;

    // Get textures from map
    sf::Texture tSkyBack, tSkyFront, tCircuit, tMap;
    Map::skyTextures(player, tSkyBack, tSkyFront);
    Map::circuitTexture(player, tCircuit);
    Map::mapTexture(tMap);

    // Create sprites and scale them accordingly
    sf::Sprite skyBack(tSkyBack), skyFront(tSkyFront), circuit(tCircuit),
        map(tMap);
    sf::Vector2u windowSize = game.getWindow().getSize();
    float backFactor = windowSize.x / (float)tSkyBack.getSize().x;
    float frontFactor = windowSize.x / (float)tSkyFront.getSize().x;
    skyBack.setScale(backFactor, backFactor);
    skyFront.setScale(frontFactor, frontFactor);

    // Sort them correctly in Y position and draw
    float currentHeight = 0;
    skyBack.setPosition(0.0f, currentHeight);
    skyFront.setPosition(0.0f, currentHeight);
    window.draw(skyBack);
    window.draw(skyFront);
    currentHeight += windowSize.y * Map::SKY_HEIGHT_PCT;
    circuit.setPosition(0.0f, currentHeight);
    window.draw(circuit);

    // Lakitu shadow
    Lakitu::drawShadow(window);

    // Circuit objects (must be before minimap)
    std::vector<std::pair<float, sf::Sprite*>> wallObjects;
    Map::getWallDrawables(window, player, scale, wallObjects);
    Map::getItemDrawables(window, player, scale, wallObjects);
    Map::getDriverDrawables(window, player, drivers, scale, wallObjects);
    player->getDrawables(window, scale, wallObjects);
    std::sort(wallObjects.begin(), wallObjects.end(),
              [](const std::pair<float, sf::Sprite*>& lhs,
                 const std::pair<float, sf::Sprite*>& rhs) {
                  return lhs.first > rhs.first;
              });
    for (const auto& pair : wallObjects) {
        window.draw(*pair.second);
    }

    // Particles
    if (player->height == 0.0f &&
        player->speedForward > player->vehicle->maxNormalLinearSpeed / 4) {
        bool small = player->animator.smallTime.asSeconds() > 0 ||
                     player->animator.smashTime.asSeconds() > 0;
        player->animator.drawParticles(window, player->getSprite(), small,
                                       player->position);
    }

    // Minimap
    currentHeight += windowSize.y * Map::CIRCUIT_HEIGHT_PCT;
    map.setPosition(0.0f, currentHeight);
    window.draw(map);

    // Minimap drivers
    std::sort(miniDrivers.begin(), miniDrivers.end(),
              [](const DriverPtr& lhs, const DriverPtr& rhs) {
                  return lhs->position.y < rhs->position.y;
              });
    for (const DriverPtr& driver : miniDrivers) {
        if (!driver->isVisible() && driver != player) continue;
        sf::Sprite miniDriver = driver->animator.getMinimapSprite(
            driver->posAngle + driver->speedTurn * 0.2f, scale);
        sf::Vector2f mapPosition = Map::mapCoordinates(driver->position);
        miniDriver.setOrigin(miniDriver.getLocalBounds().width / 2.0f,
                             miniDriver.getLocalBounds().height * 0.9f);
        miniDriver.setPosition(mapPosition.x * windowSize.x,
                               mapPosition.y * windowSize.y);
        miniDriver.scale(0.5f, 0.5f);
        window.draw(miniDriver);
    }

    // On top of the circuit, draw lakitu
    Lakitu::draw(window);

    // Draw Gui
    float pctGB = fmaxf(
        0.0f, (waitForPCTime - currentTime) * 255.0f / WAIT_FOR_PC_LAST_PLACE);
    Gui::draw(window, waitForPCTime == sf::Time::Zero
                          ? sf::Color::White
                          : sf::Color(255, pctGB, pctGB));

    // end ranks after lakitu
    EndRanks::draw(window);
}
