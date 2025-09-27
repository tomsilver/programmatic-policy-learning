"""Simple test for primitive registry system."""

from programmatic_policy_learning.lpp.dsl.primitive_registry import PrimitiveRegistry


def test_primitive_registry() -> None:
    """Simple test to verify primitive registry works."""

    # Create registry instance
    registry = PrimitiveRegistry()

    # Test 1: List available providers
    providers = registry.list_available_providers()
    print(f"Available providers: {providers}")
    assert "ggg" in providers, "GGG provider should be available"

    # Test 2: Register GGG primitives
    config = {"provider": "ggg"}
    ggg_primitives = registry.register_primitives(config)

    # Verify key primitives exist
    assert "cell_is_value" in ggg_primitives, "cell_is_value should exist"
    assert "out_of_bounds" in ggg_primitives, "out_of_bounds should exist"
