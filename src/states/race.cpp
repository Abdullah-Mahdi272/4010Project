#include "race.h"
#include <cstdio>  // for setvbuf, fflush

// #define DEBUG_POSITION_RANKING  // uncomment to show positions in ranking

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
    // FORCE UNBUFFERED STDOUT/STDERR
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    
    pushedPauseThisFrame = false;
    StateRace::currentTime = sf::Time::Zero;
    nextItemCheck = sf::Time::Zero;
    waitForPCTime = sf::Time::Zero;
    splitsInitialized = false;

    // Random agent (for testing - set to false for RL mode)
    random_enabled = false;
    randomAI = std::make_unique<RandomAgent>();

    // ALWAYS enable RL mode for training
    rlMode = true;

    // Ensure we have an Agent instance for this race
    if (agent == nullptr) {
        agent = new Agent();
    }

    // Output track info (use AI system for max gradient)
    int maxGradient = AIGradientDescent::MAX_POSITION_MATRIX;
    agent->outputTrackInfo(maxGradient, 1920, 1080);
    fflush(stdout);
    
    // Signal that episode is starting
    agent->outputEpisodeStart();
    fflush(stdout);
    
    // Output initial state immediately
    agent->updatePosition(player->position.x, player->position.y);
    agent->updateSpeed(player->speedForward, player->speedTurn);
    agent->updateAngle(player->posAngle);
    agent->updateGameState(
        player->getLastGradient() != -1 ? player->getLastGradient() : 0,
        player->getLaps(),
        player->getRank(),
        player->getCoins()
    );
    agent->outputState(0.0f, 0);
    fflush(stdout);
}

void StateRace::handleEvent(const sf::Event& event) {
    // Items
    if (Input::pressed(Key::ITEM_FRONT, event) && player->canDrive()) {
        Item::useItem(player, positions, true);
    }
    if (Input::pressed(Key::ITEM_BACK, event) && player->canDrive()) {
        Item::useItem(player, positions, false);
    }

    // Drifting
    if (Input::pressed(Key::DRIFT, event) && player->canDrive() &&
        !driftPressed) {
        driftPressed = true;
        player->shortJump();
    }
    if (Input::released(Key::DRIFT, event)) {
        driftPressed = false;
    }

    // Pause menu
    if (Input::pressed(Key::PAUSE, event) && !pushedPauseThisFrame) {
        pushedPauseThisFrame = true;
        // Call draw and store so we can draw it over the screen
        sf::RenderTexture render;
        sf::Vector2u windowSize = game.getWindow().getSize();
        render.create(windowSize.x, windowSize.y);
        fixedUpdate(sf::Time::Zero);
        draw(render);
        game.pushState(StatePtr(new StateRacePause(game, render)));
    }
}

bool StateRace::fixedUpdate(const sf::Time& deltaTime) {
    // Don't update if we already popped
    if (raceFinished) {
        return true;
    }

    // Update global time
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

        // Random agent (for testing/debugging)
        if (random_enabled && driver == player) {
            // Update randomAI for this frame, get new random action
            randomAI->update(deltaTime.asSeconds());
            // Get the random action
            const auto& d = randomAI->action();

            // If player is on the ground
            if (driver->height == 0.0f) {
                // If accelerate
                if (d.accel) {
                    // Accelerate smoothly until max speed
                    float f = driver->vehicle->motorAcceleration * 0.5f;
                    driver->speedForward = std::min(driver->speedForward + f * deltaTime.asSeconds(), 
                                                     driver->vehicle->maxNormalLinearSpeed);
                }
                // Braking
                if (d.brake) {
                    // Accelerate backwards 
                    float b = driver->vehicle->motorAcceleration * 0.6f;
                    driver->speedForward = std::max(driver->speedForward - b * deltaTime.asSeconds(), 0.0f);
                }
            }

            // Turning
            if (d.left && !d.right) {
                driver->speedTurn = -driver->vehicle->maxTurningAngularSpeed * 0.5f;
            } else if (d.right && !d.left) {
                driver->speedTurn = driver->vehicle->maxTurningAngularSpeed * 0.5f;
            } else {
                // Reduce turning (going back to moving straight)
                driver->speedTurn /= 1.5f;
            }

            // Drift (jump)
            if (d.drift && driver->canDrive() && driver->height == 0.0f) {
                driver->shortJump();
            }

            // Use item
            if (d.item && driver->canDrive() && driver->getPowerUp() != PowerUps::NONE) {
                Item::useItem(driver, positions, true);
            }
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

            // RL Agent integration
            if (rlMode && agent != nullptr) {
                // Update agent with current game state
                agent->updatePosition(driver->position.x, driver->position.y);
                agent->updateSpeed(driver->speedForward, driver->speedTurn);
                agent->updateAngle(driver->posAngle);
                agent->updateGameState(
                    driver->getLastGradient(),
                    driver->getLaps(),
                    driver->getRank(),
                    driver->getCoins()
                );

                // Output state every frame for better RL training (60 Hz)
                agent->outputState(currentTime.asSeconds(), 0);
                fflush(stdout);  // Force flush

                // Read action from Python
                int accel, brake, left, right;
                if (agent->readAction(accel, brake, left, right)) {
                    Input::setRLInput(accel, brake, left, right);
                }

                // Check race end conditions
                if (raceFinished || driver->getLaps() > NUM_LAPS_IN_CIRCUIT) {
                    if (!agent->isTerminated()) {
                        agent->outputRaceEnd(driver->getRank(), currentTime.asSeconds());
                        agent->setTerminated(true);
                    }
                }
            }
        }
    }

    // Check if AI should use its items
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

    // Ranking updates
    auto hasntFinishedBegin = positions.begin();
    // Don't sort drivers that have already finished the circuit
    while ((*hasntFinishedBegin)->getLaps() > NUM_LAPS_IN_CIRCUIT &&
           hasntFinishedBegin < positions.end()) {
        ++hasntFinishedBegin;
    }
    std::sort(hasntFinishedBegin, positions.end(),
              [](const Driver* lhs, const Driver* rhs) {
                  // Returns true if player A is ahead of B
                  if (lhs->getLaps() == rhs->getLaps()) {
                      return lhs->getLastGradient() < rhs->getLastGradient();
                  } else {
                      return lhs->getLaps() > rhs->getLaps();
                  }
              });
    // Find current player and update GUI
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

    // Start time counter if all 7 AI finished before the player
    if (waitForPCTime == sf::Time::Zero &&
        positions[positions.size() - 2]->getLaps() > NUM_LAPS_IN_CIRCUIT) {
        waitForPCTime = currentTime + WAIT_FOR_PC_LAST_PLACE;
    }
    
    // End the race if player has finished or all other AI have finished and the
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
        game.popState();
    }

    return true;
}

void StateRace::draw(sf::RenderTarget& window) {
    // Scale
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

    // End ranks after lakitu
    EndRanks::draw(window);
}
