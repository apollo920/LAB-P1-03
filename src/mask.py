import numpy as np

def softmax(x):
    x_shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def create_causal_mask(seq_len):
    mask = np.zeros((seq_len, seq_len))
    upper_triangle = np.triu(np.ones((seq_len, seq_len)), k=1)
    mask[upper_triangle == 1] = -np.inf
    return mask

if __name__ == "__main__":
    np.random.seed(7)

    seq_len = 5
    dk = 8

    Q = np.random.randn(seq_len, dk)
    K = np.random.randn(seq_len, dk)

    scores = Q @ K.T / np.sqrt(dk)

    M = create_causal_mask(seq_len)

    print("Mascara Causal M:")
    print(M)
    print()

    scores_masked = scores + M

    attention_weights = softmax(scores_masked)

    print("Pesos de Atencao apos Softmax com mascara:")
    print(np.round(attention_weights, 4))
    print()

    upper = np.triu(attention_weights, k=1)
    print("Triangulo superior (deve ser tudo 0.0):")
    print(np.round(upper, 6))

    assert np.allclose(upper, 0.0), "ERRO: ha probabilidades nao-zero no futuro!"
    print("\n[OK] Todas as probabilidades futuras sao estritamente 0.0")