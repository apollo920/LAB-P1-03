import numpy as np


def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    dk = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(dk) 

    if mask is not None:
        scores = scores + mask

    weights = softmax(scores)
    output = weights @ V 
    return output, weights


def cross_attention(encoder_out, decoder_state, W_Q, W_K, W_V):
    Q = decoder_state @ W_Q  
    K = encoder_out @ W_K    
    V = encoder_out @ W_V    
    output, weights = scaled_dot_product_attention(Q, K, V, mask=None)
    return output, weights


if __name__ == "__main__":
    np.random.seed(42)

    batch_size = 1
    seq_len_encoder = 10   
    seq_len_decoder = 4    
    d_model = 512
    dk = 64                

    encoder_output = np.random.randn(batch_size, seq_len_encoder, d_model)
    decoder_state  = np.random.randn(batch_size, seq_len_decoder,  d_model)

    print(f"encoder_output shape: {encoder_output.shape}")
    print(f"decoder_state  shape: {decoder_state.shape}")

    W_Q = np.random.randn(d_model, dk) * 0.01
    W_K = np.random.randn(d_model, dk) * 0.01
    W_V = np.random.randn(d_model, dk) * 0.01

    output, weights = cross_attention(encoder_output, decoder_state, W_Q, W_K, W_V)

    print(f"\nCross-Attention output shape:  {output.shape}")
    print(f"  esperado: (batch=1, seq_dec=4, dk=64)")

    print(f"\nPesos de alinhamento shape:    {weights.shape}")
    print(f"  esperado: (batch=1, seq_dec=4, seq_enc=10)")
    print(f"  (cada token gerado atende a toda a sequencia do encoder)")

    soma = weights.sum(axis=-1)
    print(f"\nSoma dos pesos por token (deve ser ~1.0):")
    print(np.round(soma, 6))

    assert np.allclose(soma, 1.0, atol=1e-5), "ERRO: pesos nao somam 1!"
    print("\n[OK] Cross-Attention funcionando corretamente.")