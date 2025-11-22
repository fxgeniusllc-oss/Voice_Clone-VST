#include "UndoManager.h"

MAEVNUndoManager::MAEVNUndoManager()
{
    // JUCE's UndoManager doesn't have setMaxNumberOfStoredUnits anymore
    // It now uses a configurable transaction limit
}

MAEVNUndoManager::~MAEVNUndoManager()
{
}

bool MAEVNUndoManager::undo()
{
    return undoManager.undo();
}

bool MAEVNUndoManager::redo()
{
    return undoManager.redo();
}

bool MAEVNUndoManager::canUndo() const
{
    return undoManager.canUndo();
}

bool MAEVNUndoManager::canRedo() const
{
    return undoManager.canRedo();
}

void MAEVNUndoManager::clearUndoHistory()
{
    undoManager.clearUndoHistory();
}

void MAEVNUndoManager::beginNewTransaction(const juce::String& actionName)
{
    undoManager.beginNewTransaction(actionName);
}
