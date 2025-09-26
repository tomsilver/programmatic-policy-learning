"""Simple test for primitive registry system."""

def test_primitive_registry():
    """Simple test to verify primitive registry works."""
    from programmatic_policy_learning.lpp.dsl.primitive_registry import PrimitiveRegistry
    
    # Create registry instance
    registry = PrimitiveRegistry()
    
    # Test 1: List available providers
    providers = registry.list_available_providers()
    print(f"Available providers: {providers}")
    assert "ggg" in providers, "GGG provider should be available"
    
    # Test 2: Register GGG primitives
    config = {"provider": "ggg"}
    ggg_primitives = registry.register_primitives(config)
    print(f"GGG primitives count: {len(ggg_primitives)}")
    print(f"Sample GGG primitives: {sorted(list(ggg_primitives.keys())[:3])}")
    
    # Verify key primitives exist
    assert "cell_is_value" in ggg_primitives, "cell_is_value should exist"
    assert "out_of_bounds" in ggg_primitives, "out_of_bounds should exist"
    
    # Test 3: Register PRBench primitives
    config = {"provider": "prbench"}
    prbench_primitives = registry.register_primitives(config)
    print(f"PRBench primitives count: {len(prbench_primitives)}")
    print(f"Sample PRBench primitives: {sorted(list(prbench_primitives.keys())[:3])}")
    
    # Verify key primitives exist
    assert "state_dimension" in prbench_primitives, "state_dimension should exist"
    assert "state_magnitude" in prbench_primitives, "state_magnitude should exist"
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_primitive_registry()