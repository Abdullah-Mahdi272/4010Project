#pragma once

#include <SFML/Graphics.hpp>
#include <array>
#include <vector>

class SplitTimer {
public:
    static constexpr int NUM_SPLITS = 8;
    
private:
    // Split checkpoint gradient values (initialized per map)
    std::array<int, NUM_SPLITS> checkpointGradients;
    int maxGradientValue;
    bool checkpointsInitialized;
    bool texturesLoaded;
    
    // Split times for current lap (in milliseconds)
    std::array<sf::Time, NUM_SPLITS> currentLapSplits;
    
    // Best split times ever recorded (in milliseconds)
    std::array<sf::Time, NUM_SPLITS> bestSplits;
    
    // Split times for previous lap
    std::array<sf::Time, NUM_SPLITS> previousLapSplits;
    
    // Which splits have been crossed this lap
    std::array<bool, NUM_SPLITS> splitsCrossed;
    
    // Next split index to cross (0-7)
    int nextSplitIndex;
    
    // Current lap number
    int currentLap;
    
    // Real-time state data for RL
    float currentSpeed;
    float currentTurnSpeed;
    float currentAngle;
    sf::Vector2f currentPosition;
    
    // RL output timing
    sf::Time lastRLOutput;
    static const sf::Time RL_OUTPUT_INTERVAL;  // 100ms = 10 Hz
    
    // Visual components
    sf::Texture digits[10];
    sf::Texture separators[3]; // ' '' and :
    sf::Vector2f scaleFactor;
    float windowScaleFactor;
    sf::Vector2u winSize;
    
    // Split display sprites (for showing most recent split time)
    std::array<sf::Sprite, 6> splitDisplayDigits; // MM:SS.mm format
    std::array<sf::Sprite, 2> splitDisplaySeparators;
    sf::Vector2f splitDisplayPosition;
    sf::Time splitDisplayTimer; // How long to show the split
    static const sf::Time SPLIT_DISPLAY_DURATION;
    bool showingSplit;
    int displayingSplitIndex;
    
    // Delta display (for showing if you're faster/slower)
    std::array<sf::Sprite, 6> deltaDisplayDigits;
    sf::Sprite deltaDisplaySeparator;
    sf::Sprite deltaSignSprite; // + or -
    sf::Texture deltaSignTextures[2]; // +/-
    sf::Vector2f deltaDisplayPosition;
    bool showingDelta;
    bool deltaIsPositive; // false = faster, true = slower
    
public:
    SplitTimer();
    
    // Initialize checkpoints based on max gradient value from the map
    void initializeCheckpoints(int maxGradient);
    
    // Check if player has crossed a split and record time
    // Also outputs RL state data at regular intervals (10 Hz)
    void update(const sf::Time& currentRaceTime, int currentGradient, int currentLapNum, 
                const sf::Time& deltaTime, const sf::Vector2f& position, 
                float speed, float turnSpeed, float angle);
    
    // Reset for new lap
    void resetForNewLap(int lapNumber);
    
    // Reset everything (new race)
    void reset();
    
    // Set window size for rendering
    void setWindowSize(sf::Vector2u s);
    
    // Draw split display
    void draw(sf::RenderTarget& window);
    
    // Get split time for specific split and lap (for debugging/display)
    sf::Time getSplitTime(int splitIndex, int lap = -1) const;
    
    // Get best split time
    sf::Time getBestSplit(int splitIndex) const;
    
    // Check if checkpoints are initialized
    inline bool isInitialized() const { return checkpointsInitialized; }
    
    // Check if textures loaded successfully
    inline bool isTexturesLoaded() const { return texturesLoaded; }
    
private:
    // Helper to format and display a split time
    void displaySplit(int splitIndex, const sf::Time& splitTime);
    
    // Helper to calculate and display delta
    void displayDelta(int splitIndex, const sf::Time& currentTime);
    
    // Update display timers
    void updateDisplay(const sf::Time& deltaTime);
    
    // Format time to digit sprites
    void formatTimeToSprites(const sf::Time& time, std::array<sf::Sprite, 6>& digitSprites, 
                            std::array<sf::Sprite, 2>& separatorSprites);
    
    // Output state data for RL system at regular intervals
    void outputStateData(const sf::Time& currentRaceTime, int currentGradient, int currentLapNum);
};
