#pragma once

#include <JuceHeader.h>

class MAEVNUndoManager
{
public:
    MAEVNUndoManager();
    ~MAEVNUndoManager();

    // Get the underlying JUCE UndoManager
    juce::UndoManager& getJuceUndoManager() { return undoManager; }

    // Perform undo
    bool undo();
    
    // Perform redo
    bool redo();
    
    // Check if undo is available
    bool canUndo() const;
    
    // Check if redo is available
    bool canRedo() const;
    
    // Clear undo history
    void clearUndoHistory();
    
    // Begin a new transaction
    void beginNewTransaction(const juce::String& actionName);

private:
    juce::UndoManager undoManager;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(MAEVNUndoManager)
};
