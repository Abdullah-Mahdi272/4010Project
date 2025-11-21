#include "splittimer.h"
#include <cmath>
#include <iostream>
#include <iomanip>

const sf::Time SplitTimer::SPLIT_DISPLAY_DURATION = sf::seconds(3.0f);
const sf::Time SplitTimer::RL_OUTPUT_INTERVAL = sf::milliseconds(100);  // 10 Hz

SplitTimer::SplitTimer() {
    // Initialize all member variables first
    scaleFactor = sf::Vector2f(1.5f, 1.5f);
    windowScaleFactor = 1.0f;
    winSize = sf::Vector2u(0, 0);
    
    checkpointsInitialized = false;
    texturesLoaded = false;
    maxGradientValue = 0;
    nextSplitIndex = 0;
    currentLap = 0;
    showingSplit = false;
    showingDelta = false;
    displayingSplitIndex = -1;
    deltaIsPositive = false;
    splitDisplayTimer = sf::Time::Zero;
    lastRLOutput = sf::Time::Zero;
    
    // Initialize state data
    currentSpeed = 0.0f;
    currentTurnSpeed = 0.0f;
    currentAngle = 0.0f;
    currentPosition = sf::Vector2f(0.0f, 0.0f);
    
    // Initialize arrays
    for (int i = 0; i < NUM_SPLITS; i++) {
        checkpointGradients[i] = 0;
        currentLapSplits[i] = sf::Time::Zero;
        bestSplits[i] = sf::Time::Zero;
        previousLapSplits[i] = sf::Time::Zero;
        splitsCrossed[i] = false;
    }
    
    // Load digit and separator textures (same as Timer class)
    std::string spriteFile = "assets/gui/digits.png";
    
    bool loadSuccess = true;
    
    // Check if file exists and can be loaded
    for (int i = 0; i < 10; i++) {
        if (!digits[i].loadFromFile(spriteFile, sf::IntRect(0 + (i * 9), 0, 8, 14))) {
            std::cerr << "ERROR: Failed to load digit texture " << i << " from " << spriteFile << std::endl;
            loadSuccess = false;
            break;
        }
    }
    
    if (loadSuccess) {
        for (int i = 0; i < 2; i++) {
            if (!separators[i].loadFromFile(spriteFile, sf::IntRect(90 + (i * 9), 0, 8, 14))) {
                std::cerr << "ERROR: Failed to load separator texture " << i << " from " << spriteFile << std::endl;
                loadSuccess = false;
                break;
            }
        }
    }
    
    if (loadSuccess) {
        if (!separators[2].loadFromFile(spriteFile, sf::IntRect(126, 0, 8, 14))) {
            std::cerr << "ERROR: Failed to load colon texture from " << spriteFile << std::endl;
            loadSuccess = false;
        }
    }
    
    // Load +/- signs
    if (loadSuccess) {
        if (!deltaSignTextures[0].loadFromFile(spriteFile, sf::IntRect(117, 0, 8, 14))) {
            std::cerr << "ERROR: Failed to load minus sign from " << spriteFile << std::endl;
            loadSuccess = false;
        }
    }
    
    if (loadSuccess) {
        if (!deltaSignTextures[1].loadFromFile(spriteFile, sf::IntRect(108, 0, 8, 14))) {
            std::cerr << "ERROR: Failed to load plus sign from " << spriteFile << std::endl;
            loadSuccess = false;
        }
    }
    
    if (!loadSuccess) {
        std::cerr << "CRITICAL: SplitTimer texture loading failed." << std::endl;
        std::cerr << "This is likely a working directory issue. Make sure you run the executable from the project root." << std::endl;
        std::cerr << "SplitTimer will be disabled." << std::endl;
        texturesLoaded = false;
        checkpointsInitialized = false;
        return;
    }
    
    // Mark textures as successfully loaded
    texturesLoaded = true;
    
    // Initialize sprite arrays with textures
    for (int i = 0; i < 6; i++) {
        splitDisplayDigits[i].setTexture(digits[0]);
        splitDisplayDigits[i].scale(scaleFactor);
        deltaDisplayDigits[i].setTexture(digits[0]);
        deltaDisplayDigits[i].scale(scaleFactor);
    }
    
    for (int i = 0; i < 2; i++) {
        splitDisplaySeparators[i].setTexture(separators[0]);
        splitDisplaySeparators[i].scale(scaleFactor);
    }
    
    deltaDisplaySeparator.setTexture(separators[1]);
    deltaDisplaySeparator.scale(scaleFactor);
    deltaSignSprite.setTexture(deltaSignTextures[0]);
    deltaSignSprite.scale(scaleFactor);
    
    // std::cout << "SplitTimer initialized successfully." << std::endl;
}

void SplitTimer::initializeCheckpoints(int maxGradient) {
    if (!texturesLoaded) {
        std::cerr << "WARNING: Cannot initialize checkpoints - textures not loaded." << std::endl;
        return;
    }
    
    if (maxGradient <= 0) {
        std::cerr << "ERROR: Invalid maxGradient value: " << maxGradient << std::endl;
        return;
    }
    
    maxGradientValue = maxGradient;
    
    // Calculate equal sections - gradient DECREASES as you progress
    float sectionLength = static_cast<float>(maxGradient) / static_cast<float>(NUM_SPLITS);
    
    // Set checkpoint gradient values (HIGH to LOW)
    for (int i = 0; i < NUM_SPLITS; i++) {
        checkpointGradients[i] = static_cast<int>(maxGradient - (sectionLength * (i + 1)));
    }
    
    checkpointsInitialized = true;
    
    // Debug output
    // std::cout << "Split checkpoints initialized with max gradient: " << maxGradient << std::endl;
    for (int i = 0; i < NUM_SPLITS; i++) {
        // std::cout << "  Checkpoint " << (i + 1) << ": " << checkpointGradients[i] << std::endl;
    }
}

void SplitTimer::update(const sf::Time& currentRaceTime, int currentGradient, int currentLapNum, 
                        const sf::Time& deltaTime, const sf::Vector2f& position, 
                        float speed, float turnSpeed, float angle) {
    if (!checkpointsInitialized || !texturesLoaded) {
        return;
    }
    
    // Store current state data
    currentPosition = position;
    currentSpeed = speed;
    currentTurnSpeed = turnSpeed;
    currentAngle = angle;
    
    // OUTPUT RL STATE AT REGULAR INTERVALS (10 Hz)
    // if (currentRaceTime - lastRLOutput >= RL_OUTPUT_INTERVAL) {
    //     outputStateData(currentRaceTime, currentGradient, currentLapNum);
    //     lastRLOutput = currentRaceTime;
    // }
    
    // Check if we've moved to a new lap
    if (currentLapNum != currentLap && currentLapNum > currentLap) {
        resetForNewLap(currentLapNum);
    }
    
    // Check if we've crossed the next split
    if (nextSplitIndex < NUM_SPLITS && !splitsCrossed[nextSplitIndex]) {
        if (currentGradient <= checkpointGradients[nextSplitIndex]) {
            // Record the split time
            currentLapSplits[nextSplitIndex] = currentRaceTime;
            splitsCrossed[nextSplitIndex] = true;
            
            // std::cout << "*** SPLIT " << (nextSplitIndex + 1) << " CROSSED! ***" << std::endl;
            // std::cout << "    Time: " << currentRaceTime.asSeconds() << "s" << std::endl;
            // std::cout << "    Gradient: " << currentGradient << std::endl;
            // std::cout << "    Checkpoint threshold: " << checkpointGradients[nextSplitIndex] << std::endl;
            
            // Display the split
            displaySplit(nextSplitIndex, currentRaceTime);
            
            // Calculate and display delta if we have comparison data
            if (currentLap > 1 && previousLapSplits[nextSplitIndex] != sf::Time::Zero) {
                displayDelta(nextSplitIndex, currentRaceTime);
            } else if (bestSplits[nextSplitIndex] != sf::Time::Zero) {
                displayDelta(nextSplitIndex, currentRaceTime);
            }
            
            // Move to next split
            nextSplitIndex++;
        }
    }
    
    // Update display timers
    updateDisplay(deltaTime);
}

void SplitTimer::outputStateData(const sf::Time& currentRaceTime, int currentGradient, int currentLapNum) {
    // Output in structured format for RL system to parse
    // Format: RL_STATE|time|gradient|lap|split|posX|posY|speed|turnSpeed|angle
//     std::cout << "RL_STATE|"
//               << std::fixed << std::setprecision(3)
//               << currentRaceTime.asSeconds() << "|"
//               << currentGradient << "|"
//               << currentLapNum << "|"
//               << nextSplitIndex << "|"
//               << currentPosition.x << "|"
//               << currentPosition.y << "|"
//               << currentSpeed << "|"
//               << currentTurnSpeed << "|"
//               << currentAngle
//               << std::endl;
}

void SplitTimer::resetForNewLap(int lapNumber) {
    if (!texturesLoaded) {
        return;
    }
    
    // std::cout << "=== LAP TRANSITION: " << currentLap << " -> " << lapNumber << " ===" << std::endl;
    
    // Save current lap splits to previous lap
    for (int i = 0; i < NUM_SPLITS; i++) {
        previousLapSplits[i] = currentLapSplits[i];
        
        // Update best splits if applicable
        if (currentLapSplits[i] != sf::Time::Zero) {
            if (bestSplits[i] == sf::Time::Zero || currentLapSplits[i] < bestSplits[i]) {
                bestSplits[i] = currentLapSplits[i];
                // std::cout << "  New best for split " << (i + 1) << ": " << bestSplits[i].asSeconds() << "s" << std::endl;
            }
        }
        
        // Reset current lap splits
        currentLapSplits[i] = sf::Time::Zero;
        splitsCrossed[i] = false;
    }
    
    currentLap = lapNumber;
    nextSplitIndex = 0;
    
    // std::cout << "SplitTimer reset for lap " << lapNumber << std::endl;
}

void SplitTimer::reset() {
    for (int i = 0; i < NUM_SPLITS; i++) {
        currentLapSplits[i] = sf::Time::Zero;
        bestSplits[i] = sf::Time::Zero;
        previousLapSplits[i] = sf::Time::Zero;
        splitsCrossed[i] = false;
    }
    
    nextSplitIndex = 0;
    currentLap = 0;
    showingSplit = false;
    showingDelta = false;
    splitDisplayTimer = sf::Time::Zero;
    displayingSplitIndex = -1;
    lastRLOutput = sf::Time::Zero;
    
    // Reset state data
    currentSpeed = 0.0f;
    currentTurnSpeed = 0.0f;
    currentAngle = 0.0f;
    currentPosition = sf::Vector2f(0.0f, 0.0f);
    
    // std::cout << "SplitTimer fully reset." << std::endl;
}

void SplitTimer::setWindowSize(sf::Vector2u s) {
    winSize = s;
    
    if (winSize.x == 0) {
        std::cerr << "WARNING: Window size width is 0" << std::endl;
        return;
    }
    
    const float BASIC_WIDTH = 512.0f;
    windowScaleFactor = winSize.x / BASIC_WIDTH;
    
    if (!texturesLoaded) {
        return;
    }
    
    // Scale all sprites
    for (int i = 0; i < 6; i++) {
        splitDisplayDigits[i].setScale(scaleFactor.x * windowScaleFactor, 
                                       scaleFactor.y * windowScaleFactor);
        deltaDisplayDigits[i].setScale(scaleFactor.x * windowScaleFactor, 
                                       scaleFactor.y * windowScaleFactor);
    }
    
    for (int i = 0; i < 2; i++) {
        splitDisplaySeparators[i].setScale(scaleFactor.x * windowScaleFactor, 
                                           scaleFactor.y * windowScaleFactor);
    }
    
    deltaDisplaySeparator.setScale(scaleFactor.x * windowScaleFactor, 
                                   scaleFactor.y * windowScaleFactor);
    deltaSignSprite.setScale(scaleFactor.x * windowScaleFactor, 
                            scaleFactor.y * windowScaleFactor);
    
    // Position split display in top-left area (below timer)
    splitDisplayPosition = sf::Vector2f(s.x * 0.05f, s.y * 0.15f);
    
    // Position delta display right below split display
    deltaDisplayPosition = sf::Vector2f(s.x * 0.05f, s.y * 0.22f);
    
    // std::cout << "SplitTimer window size set: " << s.x << "x" << s.y << std::endl;
}

void SplitTimer::displaySplit(int splitIndex, const sf::Time& splitTime) {
    if (!texturesLoaded) {
        return;
    }
    
    // std::cout << "*** DISPLAYING SPLIT " << (splitIndex + 1) << " on screen ***" << std::endl;
    // std::cout << "    Split time: " << splitTime.asSeconds() << "s" << std::endl;
    
    showingSplit = true;
    displayingSplitIndex = splitIndex;
    splitDisplayTimer = SPLIT_DISPLAY_DURATION;
    
    formatTimeToSprites(splitTime, splitDisplayDigits, splitDisplaySeparators);
}

void SplitTimer::displayDelta(int splitIndex, const sf::Time& currentTime) {
    if (!texturesLoaded) {
        return;
    }
    
    // Compare to best split if available, otherwise compare to previous lap
    sf::Time comparisonTime = bestSplits[splitIndex];
    if (comparisonTime == sf::Time::Zero) {
        comparisonTime = previousLapSplits[splitIndex];
    }
    
    if (comparisonTime == sf::Time::Zero) {
        showingDelta = false;
        return;
    }
    
    showingDelta = true;
    
    // Calculate delta
    sf::Time delta = currentTime - comparisonTime;
    deltaIsPositive = delta.asMilliseconds() >= 0;
    
    // std::cout << "    Delta: " << (deltaIsPositive ? "+" : "-") << std::abs(delta.asMilliseconds()) << "ms" << std::endl;
    
    // Format absolute value of delta
    sf::Time absDelta = sf::milliseconds(std::abs(delta.asMilliseconds()));
    
    // Set the sign sprite
    deltaSignSprite.setTexture(deltaIsPositive ? deltaSignTextures[1] : deltaSignTextures[0]);
    
    // Format time
    long timeAsMilli = absDelta.asMilliseconds();
    int seconds = timeAsMilli / 1000;
    timeAsMilli -= seconds * 1000;
    int millis = timeAsMilli / 10;
    
    // Bounds checking
    if (seconds > 99) seconds = 99;
    if (millis > 99) millis = 99;
    
    // Format as SS.mm
    deltaDisplayDigits[0].setTexture(digits[seconds / 10]);
    deltaDisplayDigits[1].setTexture(digits[seconds % 10]);
    deltaDisplayDigits[2].setTexture(digits[millis / 10]);
    deltaDisplayDigits[3].setTexture(digits[millis % 10]);
}

void SplitTimer::updateDisplay(const sf::Time& deltaTime) {
    if (showingSplit) {
        splitDisplayTimer -= deltaTime;
        if (splitDisplayTimer <= sf::Time::Zero) {
            showingSplit = false;
            showingDelta = false;
        }
    }
}

void SplitTimer::formatTimeToSprites(const sf::Time& time, 
                                     std::array<sf::Sprite, 6>& digitSprites,
                                     std::array<sf::Sprite, 2>& separatorSprites) {
    if (!texturesLoaded) {
        return;
    }
    
    long timeAsMilli = time.asMilliseconds();
    int minutes = timeAsMilli / 60000;
    timeAsMilli -= minutes * 60000;
    int seconds = timeAsMilli / 1000;
    timeAsMilli -= seconds * 1000;
    int millis = timeAsMilli / 10;
    
    // Bounds checking
    if (minutes > 99) minutes = 99;
    if (seconds > 59) seconds = 59;
    if (millis > 99) millis = 99;
    
    digitSprites[0].setTexture(digits[minutes / 10]);
    digitSprites[1].setTexture(digits[minutes % 10]);
    digitSprites[2].setTexture(digits[seconds / 10]);
    digitSprites[3].setTexture(digits[seconds % 10]);
    digitSprites[4].setTexture(digits[millis / 10]);
    digitSprites[5].setTexture(digits[millis % 10]);
    
    separatorSprites[0].setTexture(separators[0]); // '
    separatorSprites[1].setTexture(separators[1]); // ''
}

void SplitTimer::draw(sf::RenderTarget& window) {
    // Guard against drawing with invalid textures
    if (!texturesLoaded || !showingSplit) {
        return;
    }
    
    // Calculate fade effect in last 0.5 seconds
    sf::Color color = sf::Color::White;
    if (splitDisplayTimer.asSeconds() < 0.5f) {
        int alpha = static_cast<int>(255 * (splitDisplayTimer.asSeconds() / 0.5f));
        alpha = std::max(0, std::min(255, alpha)); // Clamp to valid range
        color.a = alpha;
    }
    
    // Draw split number label
    int separationPixels = static_cast<int>(2 * windowScaleFactor);
    int xSizeSprite = static_cast<int>(splitDisplayDigits[0].getGlobalBounds().width);
    float xPos = splitDisplayPosition.x;
    float yPos = splitDisplayPosition.y;
    
    // Draw "MM'SS''mm" format
    int digitIndex = 0;
    for (int i = 0; i < 8; i++) {
        if (i == 2) {  // first separator
            splitDisplaySeparators[0].setPosition(xPos, yPos);
            splitDisplaySeparators[0].setColor(color);
            window.draw(splitDisplaySeparators[0]);
        } else if (i == 5) {  // second separator
            splitDisplaySeparators[1].setPosition(xPos, yPos);
            splitDisplaySeparators[1].setColor(color);
            window.draw(splitDisplaySeparators[1]);
        } else {  // digit
            if (digitIndex < 6) {  // Bounds check
                splitDisplayDigits[digitIndex].setPosition(xPos, yPos);
                splitDisplayDigits[digitIndex].setColor(color);
                window.draw(splitDisplayDigits[digitIndex]);
                digitIndex++;
            }
        }
        xPos += xSizeSprite + separationPixels;
    }
    
    // Draw delta if available
    if (showingDelta) {
        xPos = deltaDisplayPosition.x;
        yPos = deltaDisplayPosition.y;
        
        // Color based on whether you're faster or slower
        sf::Color deltaColor = deltaIsPositive ? 
            sf::Color(255, 100, 100, color.a) : 
            sf::Color(100, 255, 100, color.a);
        
        // Draw sign
        deltaSignSprite.setPosition(xPos, yPos);
        deltaSignSprite.setColor(deltaColor);
        window.draw(deltaSignSprite);
        xPos += xSizeSprite + separationPixels;
        
        // Draw SS
        deltaDisplayDigits[0].setPosition(xPos, yPos);
        deltaDisplayDigits[0].setColor(deltaColor);
        window.draw(deltaDisplayDigits[0]);
        xPos += xSizeSprite + separationPixels;
        
        deltaDisplayDigits[1].setPosition(xPos, yPos);
        deltaDisplayDigits[1].setColor(deltaColor);
        window.draw(deltaDisplayDigits[1]);
        xPos += xSizeSprite + separationPixels;
        
        // Draw decimal separator
        deltaDisplaySeparator.setPosition(xPos, yPos);
        deltaDisplaySeparator.setColor(deltaColor);
        window.draw(deltaDisplaySeparator);
        xPos += xSizeSprite + separationPixels;
        
        // Draw mm
        deltaDisplayDigits[2].setPosition(xPos, yPos);
        deltaDisplayDigits[2].setColor(deltaColor);
        window.draw(deltaDisplayDigits[2]);
        xPos += xSizeSprite + separationPixels;
        
        deltaDisplayDigits[3].setPosition(xPos, yPos);
        deltaDisplayDigits[3].setColor(deltaColor);
        window.draw(deltaDisplayDigits[3]);
    }
}

sf::Time SplitTimer::getSplitTime(int splitIndex, int lap) const {
    if (splitIndex < 0 || splitIndex >= NUM_SPLITS) {
        return sf::Time::Zero;
    }
    
    if (lap == -1 || lap == currentLap) {
        return currentLapSplits[splitIndex];
    } else if (lap == currentLap - 1) {
        return previousLapSplits[splitIndex];
    }
    
    return sf::Time::Zero;
}

sf::Time SplitTimer::getBestSplit(int splitIndex) const {
    if (splitIndex < 0 || splitIndex >= NUM_SPLITS) {
        return sf::Time::Zero;
    }
    return bestSplits[splitIndex];
}
