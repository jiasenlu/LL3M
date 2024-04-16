
# import pickle
# # pickle.dump({'input_ids':input_ids, 'images':images, 'image_input_idx':image_input_idx}, open('save.pkl', 'wb'))
# aa = pickle.load(open('input.pkl', 'rb'))
# input_ids = aa['input_ids']
# pixel_values = aa['pixel_values']
# pixel_values = np.transpose(pixel_values, (0,2,3,1))
# images = einops.rearrange(
#     pixel_values, 'b (dy h dh) (dx w dw) c -> b (dy dx) (h w) (dh dw c)',
#     dh=14,
#     dw=14,
#     dy=1,
#     dx=1,
#     h=24,
#     w=24
# )
# images = jnp.repeat(images, batch_size, 0) 

# input_ids = jnp.concat([input_ids[:,:1], input_ids[:,1:2].repeat(576,1), input_ids[:,2:]], axis=-1)
# input_ids = jnp.repeat(input_ids, batch_size, 0)

# attention_mask = jnp.ones(input_ids.shape, jnp.int32)
# position_ids = jnp.repeat(jnp.arange(input_ids.shape[1])[None,:], batch_size, 0)
# image_input_idx =  jnp.repeat(jnp.arange(1, 577)[None,:], batch_size, 0)
        

        
