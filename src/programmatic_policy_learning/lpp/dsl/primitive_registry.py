"""Primitive registry for managing provider-specific primitive sets."""

import inspect
from typing import Any, Callable, Dict

import importlib

def _get_module_functions(module) -> Dict[str, Callable]:
    """Helper to extract all public functions from a module."""
    return {
        name: func for name, func in inspect.getmembers(module, inspect.isfunction)
        if not name.startswith('_')  # Exclude private functions
    }


def create_primitives(provider_name: str) -> Dict[str, Callable]:
    """Generic function to create primitives for any provider.
    
    Args:
        provider_name: Name of the provider (e.g., 'ggg', 'prbench', 'gymnasium')
        
    Returns:
        Dictionary mapping primitive names to functions
        
    Raises:
        ImportError: If the provider module cannot be imported
    """
    try:
        # Dynamically import the provider module
        module_path = f"programmatic_policy_learning.lpp.dsl.providers.{provider_name}_primitives"
        provider_module = importlib.import_module(module_path)
        
        # Extract all public functions from the module
        return _get_module_functions(provider_module)
        
    except ImportError as e:
        raise ImportError(f"Could not import primitives for provider '{provider_name}': {e}") from e


class PrimitiveRegistry:
    """Registry for primitive providers."""

    def __init__(self) -> None:
        # List of available providers (corresponding to *_primitives.py files)
        self._available_providers = ["ggg", "prbench"]

    def register_primitives(self, config: Any) -> Dict[str, Callable]:
        """Load primitives based on configuration.
        
        Args:
            config: Configuration object with 'provider' field
            
        Returns:
            Dictionary mapping primitive names to functions
            
        Raises:
            ValueError: If provider is not found in config or not registered
        """
        provider = getattr(config, 'provider', config.get('provider'))
        
        if provider not in self._available_providers:
            raise ValueError(f"Unknown provider '{provider}'. Available providers: {self._available_providers}")
        
        return create_primitives(provider)
    
    
    def list_available_providers(self) -> list[str]:
        """Get list of available providers."""
        return self._available_providers.copy()
    
    def register_provider(self, provider: str) -> None:
        """Register a new primitive provider.
        
        Args:
            provider: Provider name (should correspond to {provider}_primitives.py file)
        """
        if provider not in self._available_providers:
            self._available_providers.append(provider)


# Create a global instance for convenience (similar to env registry pattern)
# primitive_registry = PrimitiveRegistry()

