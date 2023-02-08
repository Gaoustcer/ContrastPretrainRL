# Contrastive Predictive Coding
Encode the context into latent space embedding, the context is defined with
$$
s_0,a_0,s_1,a_1,\cdots,s_t
$$
state and action are encoded into tensor. This consists with N+1 states and N actions. To modified relationship between context and current stragegy, we need to treat $(c_t,a_t)$ as a positive pair
min length of sequence of 145,