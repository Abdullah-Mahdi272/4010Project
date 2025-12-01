#!/bin/bash
# Comprehensive RL Mode Diagnostic Script

echo "======================================================================"
echo "MARIO KART RL MODE COMPREHENSIVE DIAGNOSTIC"
echo "======================================================================"
echo ""

GAME_DIR="/home/student/F25_4010/1.2/cloned_4010Project"
GAME_EXE="$GAME_DIR/super_mario_kart"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: File existence
echo "1. CHECKING FILE STRUCTURE"
echo "----------------------------------------------------------------------"

if [ -f "$GAME_EXE" ]; then
    echo -e "${GREEN}✓${NC} Game executable exists: $GAME_EXE"
    echo "  File size: $(du -h "$GAME_EXE" | cut -f1)"
    echo "  Modified: $(stat -c %y "$GAME_EXE" | cut -d'.' -f1)"
else
    echo -e "${RED}✗${NC} Game executable NOT FOUND: $GAME_EXE"
    exit 1
fi

if [ -d "$GAME_DIR/assets" ]; then
    echo -e "${GREEN}✓${NC} Assets directory exists"
else
    echo -e "${RED}✗${NC} Assets directory NOT FOUND"
    exit 1
fi

echo ""

# Check 2: Binary strings
echo "2. CHECKING COMPILED BINARY FOR RL STRINGS"
echo "----------------------------------------------------------------------"

RL_STRINGS=(
    "RL MODE DETECTED"
    "TRACK_INFO"
    "EPISODE_START"
    "RL_STATE"
    "outputState"
    "readAction"
    "getenv"
)

FOUND=0
TOTAL=${#RL_STRINGS[@]}

for str in "${RL_STRINGS[@]}"; do
    if strings "$GAME_EXE" | grep -q "$str"; then
        echo -e "${GREEN}✓${NC} Found: '$str'"
        ((FOUND++))
    else
        echo -e "${RED}✗${NC} Missing: '$str'"
    fi
done

echo ""
echo "Found $FOUND/$TOTAL required strings"

if [ $FOUND -lt $TOTAL ]; then
    echo -e "${YELLOW}⚠${NC} Binary is missing RL code!"
    echo "  Run: cd $GAME_DIR && make clean && make debug"
    echo ""
fi

# Check 3: Source files
echo "3. CHECKING SOURCE FILES"
echo "----------------------------------------------------------------------"

SOURCE_FILES=(
    "agent.cpp"
    "agent.h"
    "initload.cpp"
    "race.cpp"
    "input.cpp"
)

cd "$GAME_DIR"
for file in "${SOURCE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} Found: $file"
        # Check for key content
        if [ "$file" = "initload.cpp" ]; then
            if grep -q "RL_MODE" "$file"; then
                echo "    Contains RL_MODE check: YES"
            else
                echo -e "    ${RED}Contains RL_MODE check: NO${NC}"
            fi
        fi
    else
        echo -e "${RED}✗${NC} Missing: $file"
    fi
done

echo ""

# Check 4: Test RL_MODE with timeout
echo "4. TESTING RL_MODE DETECTION (10 second test)"
echo "----------------------------------------------------------------------"
echo "Starting game with RL_MODE=1..."
echo "Looking for RL mode messages..."
echo ""

cd "$GAME_DIR"

# Capture output with timeout
OUTPUT=$(timeout 10s env RL_MODE=1 "$GAME_EXE" 2>&1)

echo "--- Game Output (first 50 lines) ---"
echo "$OUTPUT" | head -n 50
echo "--- End Output ---"
echo ""

# Analyze output
echo "5. ANALYSIS"
echo "----------------------------------------------------------------------"

if echo "$OUTPUT" | grep -q "RL MODE DETECTED"; then
    echo -e "${GREEN}✓${NC} RL MODE WAS DETECTED!"
else
    echo -e "${RED}✗${NC} RL MODE NOT DETECTED"
    echo "  The game is not entering RL mode."
fi

if echo "$OUTPUT" | grep -q "INITLOAD"; then
    echo -e "${GREEN}✓${NC} Debug output from initload found"
else
    echo -e "${YELLOW}⚠${NC} No debug output from initload"
    echo "  You may need to use initload_debug.cpp"
fi

if echo "$OUTPUT" | grep -q "getenv.*RL_MODE"; then
    echo -e "${GREEN}✓${NC} getenv check found in output"
    
    if echo "$OUTPUT" | grep -q "getenv.*returned.*NULL"; then
        echo -e "${RED}  Environment variable NOT being read!${NC}"
    elif echo "$OUTPUT" | grep -q "getenv.*returned.*1"; then
        echo -e "${GREEN}  Environment variable is being read correctly${NC}"
    fi
else
    echo -e "${YELLOW}⚠${NC} getenv check not found"
fi

if echo "$OUTPUT" | grep -q "TRACK_INFO"; then
    echo -e "${GREEN}✓${NC} TRACK_INFO message found"
else
    echo -e "${RED}✗${NC} TRACK_INFO message NOT found"
fi

if echo "$OUTPUT" | grep -q "EPISODE_START"; then
    echo -e "${GREEN}✓${NC} EPISODE_START message found"
else
    echo -e "${RED}✗${NC} EPISODE_START message NOT found"
fi

if echo "$OUTPUT" | grep -q "RL_STATE"; then
    COUNT=$(echo "$OUTPUT" | grep -c "RL_STATE")
    echo -e "${GREEN}✓${NC} RL_STATE messages found ($COUNT messages)"
else
    echo -e "${RED}✗${NC} RL_STATE messages NOT found"
fi

if echo "$OUTPUT" | grep -q "SplitTimer initialized"; then
    echo -e "${YELLOW}⚠${NC} Normal game startup detected"
    echo "  This means the game is running in normal mode, not RL mode"
fi

echo ""
echo "======================================================================"
echo "SUMMARY AND RECOMMENDATIONS"
echo "======================================================================"

if echo "$OUTPUT" | grep -q "RL MODE DETECTED" && echo "$OUTPUT" | grep -q "RL_STATE"; then
    echo -e "${GREEN}✓ SUCCESS!${NC} RL mode is working!"
    echo ""
    echo "Your Python environment should now work."
    echo "Try running: python3 fixed_env.py"
else
    echo -e "${RED}✗ FAILURE${NC} - RL mode is not working properly."
    echo ""
    echo "NEXT STEPS:"
    echo ""
    
    if ! echo "$OUTPUT" | grep -q "INITLOAD"; then
        echo "1. Replace initload.cpp with debug version:"
        echo "   cp initload_debug.cpp initload.cpp"
        echo ""
    fi
    
    if [ $FOUND -lt $TOTAL ]; then
        echo "2. Recompile the game:"
        echo "   cd $GAME_DIR"
        echo "   make clean"
        echo "   make debug"
        echo ""
    fi
    
    echo "3. Run this diagnostic again to see if it's fixed"
    echo ""
    
    if echo "$OUTPUT" | grep -q "getenv.*returned.*NULL"; then
        echo "⚠ CRITICAL: Environment variable not being read!"
        echo "  This is a fundamental issue with how the game reads environment vars."
        echo "  Check that getenv() is being called correctly in initload.cpp"
    fi
fi

echo "======================================================================"
