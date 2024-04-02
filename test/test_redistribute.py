import jax
import jax.numpy as jnp

def redistribute_overflowed_tokens(router_probs, expert_capacity, num_selected_experts):
    num_groups, tokens_per_group, num_experts = router_probs.shape
    
    top_k_values, top_k_indices = jax.lax.top_k(router_probs, num_selected_experts)
    flat_indices = top_k_indices.reshape(num_groups, -1)
    one_hot_assignments = jax.nn.one_hot(flat_indices, num_experts, dtype=jnp.int32)
    # initial_load = jnp.sum(one_hot_assignments, axis=1).reshape(num_groups, tokens_per_group, num_experts)
    initial_load = jnp.cumsum(one_hot_assignments, axis=1) * one_hot_assignments - 1.0
    
    current_load = initial_load.max(1)
    overflow_mask = initial_load > expert_capacity
    overflow_tokens = jnp.any(overflow_mask, axis=-1)
    
    def redistribute(overflow_token, current_load):
        least_loaded_experts = jnp.argmin(current_load, axis=-1)
        updated_assignment = overflow_token
        # Correctly update the last element for each row with least_loaded_experts.
        # Note: Assuming overflow_token is a 2D array [tokens_per_group, num_selected_experts]
        updated_assignment = updated_assignment.at[:, -1].set(least_loaded_experts)
        return updated_assignment
    
    adjusted_expert_indices = jax.vmap(redistribute, in_axes=(0, 0))(overflow_tokens, current_load)
    
    return adjusted_expert_indices

# Mock data setup remains the same.
num_groups = 4
tokens_per_group = 80
num_experts = 8
router_probs = jax.random.uniform(jax.random.PRNGKey(0), shape=(num_groups, tokens_per_group, num_experts))
expert_capacity = 20  # Example fixed capacity.
num_selected_experts = 2  # Select top-3 experts based on routing probabilities.

# Execute the redistribution function with corrected array update.
adjusted_expert_indices = redistribute_overflowed_tokens(router_probs, expert_capacity, num_selected_experts)


import pdb; pdb.set_trace()
