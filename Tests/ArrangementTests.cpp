#include "../Source/Parser/Arrangement.h"
#include <juce_core/juce_core.h>
#include <cassert>
#include <iostream>

void testArrangementPositionTracking()
{
    std::cout << "Testing arrangement position tracking..." << std::endl;
    
    Arrangement arrangement;
    
    // Set arrangement script
    arrangement.setScript("[INTRO:0:8] [VERSE:8:16] [HOOK:24:16]");
    
    // Test position at start
    arrangement.updatePosition(0.0, 120.0, 4, 4);
    assert(arrangement.getCurrentSection() == "INTRO" && "Should be in INTRO at position 0");
    
    // Test position in VERSE
    arrangement.updatePosition(12.0, 120.0, 4, 4);
    assert(arrangement.getCurrentSection() == "VERSE" && "Should be in VERSE at position 12");
    
    // Test position in HOOK
    arrangement.updatePosition(30.0, 120.0, 4, 4);
    assert(arrangement.getCurrentSection() == "HOOK" && "Should be in HOOK at position 30");
    
    // Test position outside any section
    arrangement.updatePosition(100.0, 120.0, 4, 4);
    assert(arrangement.getCurrentSection() == "" && "Should have no section at position 100");
    
    std::cout << "✓ Position tracking test passed" << std::endl;
}

void testSectionBoundaries()
{
    std::cout << "Testing section boundaries..." << std::endl;
    
    Arrangement arrangement;
    arrangement.setScript("[INTRO:0:8] [VERSE:8:16]");
    
    // Test exactly at boundary between sections
    arrangement.updatePosition(8.0, 120.0, 4, 4);
    juce::String section = arrangement.getCurrentSection();
    assert((section == "INTRO" || section == "VERSE") && "Should be at boundary between INTRO and VERSE");
    
    // Test just before end
    arrangement.updatePosition(7.99, 120.0, 4, 4);
    assert(arrangement.getCurrentSection() == "INTRO" && "Should still be in INTRO");
    
    // Test just after start
    arrangement.updatePosition(8.01, 120.0, 4, 4);
    assert(arrangement.getCurrentSection() == "VERSE" && "Should be in VERSE");
    
    std::cout << "✓ Section boundaries test passed" << std::endl;
}

void testEmptyArrangement()
{
    std::cout << "Testing empty arrangement..." << std::endl;
    
    Arrangement arrangement;
    arrangement.setScript("");
    arrangement.updatePosition(10.0, 120.0, 4, 4);
    
    assert(arrangement.getCurrentSection() == "" && "Empty arrangement should return empty section");
    
    std::cout << "✓ Empty arrangement test passed" << std::endl;
}

void testDefaultArrangement()
{
    std::cout << "Testing default arrangement..." << std::endl;
    
    Arrangement arrangement;  // Has a default arrangement
    arrangement.updatePosition(5.0, 120.0, 4, 4);
    
    // Default arrangement should have a section at position 5
    juce::String section = arrangement.getCurrentSection();
    assert(!section.isEmpty() && "Default arrangement should have sections");
    
    std::cout << "✓ Default arrangement test passed" << std::endl;
}

int main()
{
    std::cout << "=== Running Arrangement Tests ===" << std::endl;
    
    try {
        testArrangementPositionTracking();
        testSectionBoundaries();
        testEmptyArrangement();
        testDefaultArrangement();
        
        std::cout << "\n✅ All Arrangement tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
