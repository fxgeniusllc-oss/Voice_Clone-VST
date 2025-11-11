#include "UndoManager.h"

MAEVNUndoManager::MAEVNUndoManager()
{
    undoManager.setMaxNumberOfStoredUnits(100);
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
