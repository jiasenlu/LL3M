import jax
import jax.numpy as jnp

def redistribute_overflowed_tokens(router_probs, expert_capacity, num_selected_experts):
    num_groups, tokens_per_group, num_experts = router_probs.shape
    
    # Get the top-k experts for each token
    top_k_values, top_k_indices = jax.lax.top_k(router_probs, num_selected_experts)
    flat_indices = top_k_indices.reshape(num_groups, -1)
    
    # Create one-hot vectors for expert assignments
    one_hot_assignments = jax.nn.one_hot(flat_indices, num_experts, dtype=jnp.float32)
    initial_load = jnp.sum(one_hot_assignments, axis=1)

    # Identify overloaded experts
    overflow_mask = initial_load > expert_capacity
    # Compute how much each expert is overloaded
    overflow_amount = (initial_load - expert_capacity) * overflow_mask
    
    # Compute the current load on each expert
    current_load = jnp.where(overflow_mask, expert_capacity, initial_load)
    
    available_capacity = expert_capacity - current_load
    selection_mask = available_capacity > 0
    least_loaded_experts = jnp.argmax(available_capacity * selection_mask, axis=-1)
    redistribute_amount = jnp.minimum(overflow_amount, available_capacity[jnp.arange(num_groups), least_loaded_experts][:,None])
    
    # get the mask only masked token is activated. 
    token_priority = jnp.cumsum(one_hot_assignments, axis=1) * one_hot_assignments - 1.0
    token_priority = jnp.reshape(token_priority, (num_groups, tokens_per_group, num_selected_experts, num_experts))
    overflow_token = (token_priority >= expert_capacity).sum(-2)
    
    assigned_token = ((token_priority < expert_capacity) & (token_priority >= 0)).sum(-2)
    
    # we can boost the probabilty of token that both token is missing. 
    factor = .2
    overflow_token_factor = (overflow_token.sum(-1) == num_selected_experts) * factor + jnp.array(overflow_token.sum(-1) > 0, jnp.int32)

    # remove the the probablity for assigned token. 
    router_probs_2 = (assigned_token == 0) * router_probs * overflow_token_factor[:,:,None]

    # now the new router probs are removed both for the assigned token and only for overflow toke. 
    # find the most of the capacity lies in:
    # sort the router_probs_2
    sort_prob = jnp.sort(router_probs_2, axis=-2)[:,::-1]

    max_token_prob = jnp.take_along_axis(sort_prob, jnp.array(available_capacity[:,None,:], jnp.int32), axis=1)
    overfolow_assinged_token = router_probs_2 > max_token_prob
    
    updated_assinged_token = jnp.array(overfolow_assinged_token, jnp.int32) + assigned_token
    
    # take the probablity at the threshold. 
    return updated_assinged_token

# Mock data setup
num_groups = 4
tokens_per_group = 80
num_experts = 8
router_probs = jax.random.uniform(jax.random.PRNGKey(0), shape=(num_groups, tokens_per_group, num_experts))
expert_capacity = 20  # Example fixed capacity
num_selected_experts = 2  # Select top-2 experts based on routing probabilities

# Execute the redistribution function with corrected array update
updated_assinged_token = redistribute_overflowed_tokens(router_probs, expert_capacity, num_selected_experts)
