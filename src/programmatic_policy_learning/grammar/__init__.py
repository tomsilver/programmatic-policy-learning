"""Grammar module for logical program synthesis.

This module provides tools for building and using context-free grammars
to generate logical programs for policy learning. It includes:

- GrammarBuilder: Constructs grammars for different environments
- ProgramGenerator: Generates programs using grammar rules
"""

from .grammar_builder import GrammarBuilder
from .program_generator import ProgramGenerator

__all__ = ["GrammarBuilder", "ProgramGenerator"]
