#include "../Source/Parser/ScriptParser.h"
#include <juce_core/juce_core.h>
#include <cassert>
#include <iostream>

void testBasicParsing()
{
    std::cout << "Testing basic script parsing..." << std::endl;
    
    ScriptParser parser;
    juce::String script = "[INTRO:0:8] [VERSE:8:16] [HOOK:24:16]";
    
    bool result = parser.parseScript(script);
    assert(result && "Script parsing should succeed");
    
    auto sections = parser.getSections();
    assert(sections.size() == 3 && "Should have 3 sections");
    
    // Check first section
    assert(sections[0].name == "INTRO" && "First section should be INTRO");
    assert(sections[0].startTime == 0.0 && "INTRO should start at 0");
    assert(sections[0].duration == 8.0 && "INTRO should be 8 quarters");
    
    // Check second section
    assert(sections[1].name == "VERSE" && "Second section should be VERSE");
    assert(sections[1].startTime == 8.0 && "VERSE should start at 8");
    assert(sections[1].duration == 16.0 && "VERSE should be 16 quarters");
    
    // Check third section
    assert(sections[2].name == "HOOK" && "Third section should be HOOK");
    assert(sections[2].startTime == 24.0 && "HOOK should start at 24");
    assert(sections[2].duration == 16.0 && "HOOK should be 16 quarters");
    
    std::cout << "✓ Basic parsing test passed" << std::endl;
}

void testEmptyScript()
{
    std::cout << "Testing empty script..." << std::endl;
    
    ScriptParser parser;
    juce::String emptyScript = "";
    
    bool result = parser.parseScript(emptyScript);
    assert(result && "Empty script should be valid");
    
    auto sections = parser.getSections();
    assert(sections.size() == 0 && "Empty script should have no sections");
    
    std::cout << "✓ Empty script test passed" << std::endl;
}

void testInvalidScript()
{
    std::cout << "Testing invalid script..." << std::endl;
    
    ScriptParser parser;
    juce::String invalidScript = "[INTRO:invalid:data]";
    
    bool result = parser.parseScript(invalidScript);
    assert(!result && "Invalid script should fail to parse");
    
    std::cout << "✓ Invalid script test passed" << std::endl;
}

void testComplexArrangement()
{
    std::cout << "Testing complex arrangement..." << std::endl;
    
    ScriptParser parser;
    juce::String script = "[INTRO:0:8] [VERSE:8:16] [HOOK:24:16] [808:40:8] [VERSE:48:16] [HOOK:64:16] [OUTRO:80:8]";
    
    bool result = parser.parseScript(script);
    assert(result && "Complex script should parse");
    
    auto sections = parser.getSections();
    assert(sections.size() == 7 && "Should have 7 sections");
    
    // Verify section types
    assert(sections[0].name == "INTRO" && "First should be INTRO");
    assert(sections[3].name == "808" && "Fourth should be 808");
    assert(sections[6].name == "OUTRO" && "Last should be OUTRO");
    
    std::cout << "✓ Complex arrangement test passed" << std::endl;
}

int main()
{
    std::cout << "=== Running ScriptParser Tests ===" << std::endl;
    
    try {
        testBasicParsing();
        testEmptyScript();
        testInvalidScript();
        testComplexArrangement();
        
        std::cout << "\n✅ All ScriptParser tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Test failed with unknown exception" << std::endl;
        return 1;
    }
}
